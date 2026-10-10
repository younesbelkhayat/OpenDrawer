"""Raw line extraction from the selected cartoon.

Produces a deliberately *generous* line mask at the work resolution by
combining complementary cues; the region stage then closes, prunes and
redraws clean lines:

* ``dark_strokes``     — the cartoon's own dark outlines (thin dark structures)
* ``color_boundaries`` — borders between flat color areas (k-means in LAB)
* ``neural_lineart``   — optional ControlNet lineart annotator (eyes, mouth, folds)
* ``subject_outline``  — the silhouette, so the subject is always closed

Also writes the color-flattened cartoon and the subject mask used downstream.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lib.difficulty import WORK_LONG_SIDE, get_preset
from lib.imaging import (
    disk,
    kmeans_lab,
    lab_to_rgb,
    load_mask,
    load_rgb,
    resize_long_side,
    resize_to,
    rgb_to_lab,
    save_mask,
    save_rgb,
    subsample,
)
from lib.model_registry import model_registry
from tools.base_tool import (
    BaseTool,
    Determinism,
    ResourceProfile,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)


INK_MAX_L = 26.0


def subject_mask_from(cartoon: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """Subject = the cartoon's own drawing (non-white pixels near the photo
    foreground), holes filled. Hugging the drawn outline matters: a looser
    photo silhouette would leave the cartoon's outline inside the subject and
    produce doubled lines."""
    h, w = fg.shape
    margin = max(3, int(max(h, w) * 0.015))
    nonwhite = np.any(cv2.GaussianBlur(cartoon, (5, 5), 0) < 238, axis=2)
    near = cv2.dilate(fg.astype(np.uint8), disk(margin)) > 0
    deep = cv2.erode(fg.astype(np.uint8), disk(margin * 2)) > 0  # unlined white interiors
    mask = (nonwhite & near) | deep
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, disk(max(2, margin // 2))) > 0
    # Fill interior holes (eye whites, white shirts stay inside the subject).
    inv = (~mask).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        if not (x == 0 or y == 0 or x + bw == w or y + bh == h):
            mask[labels == i] = True
    # Drop specks.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    min_area = 0.002 * h * w
    for i in range(1, n):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def flatten_colors(cartoon: np.ndarray, mask: np.ndarray, k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Quantized label map (work size) + flat RGB, computed at reduced size."""
    h, w = mask.shape
    small = resize_long_side(cartoon, 1200)
    sh, sw = small.shape[:2]
    small_mask = resize_to(mask, sw, sh)
    smooth = cv2.pyrMeanShiftFiltering(small, 8, 18)
    lab = rgb_to_lab(smooth)
    pixels = lab[small_mask] if small_mask.any() else lab.reshape(-1, 3)
    centers, _ = kmeans_lab(subsample(pixels, 40000, seed=seed), k, seed=seed)
    flat = lab.reshape(-1, 3)
    labels = np.argmin(np.linalg.norm(flat[:, None, :] - centers[None], axis=2), axis=1).reshape(sh, sw)
    labels = labels.astype(np.uint8)
    labels[~small_mask] = 255
    big = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
    big = cv2.medianBlur(big, 7)
    palette = np.vstack([lab_to_rgb(centers), np.array([[255, 255, 255]], np.uint8)])
    lut = np.full((256, 3), 255, np.uint8)
    lut[: len(centers)] = palette[: len(centers)]
    return big, lut[big]


def boundaries(labels: np.ndarray) -> np.ndarray:
    """Pixels whose right or bottom neighbor has a different label."""
    b = np.zeros(labels.shape, bool)
    b[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    b[:-1, :] |= labels[:-1, :] != labels[1:, :]
    return b


def dark_strokes(cartoon: np.ndarray, mask: np.ndarray, stroke_px: int) -> np.ndarray:
    """Thin dark structures (the cartoon's outlines), not large dark areas."""
    lab = rgb_to_lab(cartoon)
    # Ink is near-black (L* ~10-25); dark hair or fur (~30-40) must not count.
    dark = (lab[..., 0] < INK_MAX_L) & mask
    thick = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, disk(max(2, int(stroke_px * 1.5)))) > 0
    return dark & ~thick


def neural_lineart(cartoon: np.ndarray, commercial_safe: bool) -> np.ndarray:
    """ControlNet lineart annotator → bool line mask (raises when unavailable)."""
    from controlnet_aux import LineartDetector
    from PIL import Image

    path = model_registry.path("lineart_annotators", commercial_safe)
    detector = LineartDetector.from_pretrained(str(path))
    out = detector(Image.fromarray(cartoon), detect_resolution=1024, image_resolution=1024, coarse=False)
    gray = np.array(out.convert("L"))
    gray = cv2.resize(gray, (cartoon.shape[1], cartoon.shape[0]), interpolation=cv2.INTER_LINEAR)
    lines = gray > 110 if gray.mean() < 127 else gray < 145
    return cv2.morphologyEx(lines.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0


class LineartExtract(BaseTool):
    name = "lineart_extract"
    version = "1.0.0"
    tier = ToolTier.ANALYZE
    capability = "line_art"
    provider = "opendrawer"
    stability = ToolStability.BETA
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.HYBRID

    dependencies = ["python:cv2", "python:numpy"]
    install_instructions = (
        "pip install opencv-python-headless numpy\n"
        "Optionnel (traits du visage plus fins) : pip install controlnet-aux torch && "
        "python -m opendrawer models download lineart_annotators"
    )
    capabilities = ["line_extraction", "color_flattening", "subject_mask"]
    best_for = ["cartoons à aplats et contours nets"]
    required_models = ["lineart_annotators"]

    input_schema = {
        "type": "object",
        "required": ["cartoon_path", "foreground_mask_path", "output_dir"],
        "properties": {
            "cartoon_path": {"type": "string"},
            "foreground_mask_path": {"type": "string", "description": "photo_analysis.foreground_mask_path"},
            "output_dir": {"type": "string"},
            "difficulty": {"type": "string", "default": "medium"},
            "neural": {"type": "string", "enum": ["auto", "on", "off"], "default": "auto"},
            "commercial_safe": {"type": "boolean", "default": False},
            "work_long_side": {"type": "integer", "default": WORK_LONG_SIDE},
            "seed": {"type": "integer", "default": 0},
        },
    }
    resource_profile = ResourceProfile(cpu_cores=2, ram_mb=2500)
    idempotency_key_fields = ["cartoon_path", "difficulty", "neural", "work_long_side"]

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        started = time.monotonic()
        try:
            preset = get_preset(inputs.get("difficulty"))
            cartoon_src = load_rgb(inputs["cartoon_path"])
            fg_src = load_mask(inputs["foreground_mask_path"])
        except Exception as exc:
            return ToolResult(success=False, error=f"lineart_extract: invalid inputs: {exc}")

        long_side = int(inputs.get("work_long_side", WORK_LONG_SIDE))
        cartoon = resize_long_side(cartoon_src, long_side)
        h, w = cartoon.shape[:2]
        fg = resize_to(fg_src, w, h)
        out_dir = Path(inputs["output_dir"])
        warnings: list[str] = []

        subject = subject_mask_from(cartoon, fg)
        if not subject.any():
            return ToolResult(success=False, error="lineart_extract: empty subject mask")

        methods = []
        labels, flat = flatten_colors(cartoon, subject, preset.quant_colors, int(inputs.get("seed", 0)))
        color_b = boundaries(labels) & subject
        methods.append("color_boundaries")
        strokes = dark_strokes(cartoon, subject, preset.line_px)
        methods.append("dark_strokes")
        drawn = strokes.copy()  # strokes actually drawn by the illustrator
        outline = cv2.morphologyEx(subject.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        methods.append("subject_outline")
        lines = color_b | strokes | outline

        neural_mode = inputs.get("neural", "auto")
        if neural_mode != "off":
            try:
                neural = neural_lineart(cartoon, bool(inputs.get("commercial_safe")))
                lines |= neural & subject
                drawn |= neural & subject
                methods.append("neural_lineart")
            except Exception as exc:
                if neural_mode == "on":
                    return ToolResult(success=False, error=f"neural lineart unavailable: {exc}")
                warnings.append(f"neural lineart skipped ({type(exc).__name__})")

        line_path = out_dir / "line_mask.png"
        strokes_path = out_dir / "strokes_mask.png"
        flat_path = out_dir / "flat.png"
        subject_path = out_dir / "subject_mask.png"
        # Store lines as black-on-white so the file is directly viewable.
        save_mask(~lines, line_path)
        save_mask(~drawn, strokes_path)
        save_rgb(flat, flat_path)
        save_mask(subject, subject_path)
        work_cartoon = save_rgb(cartoon, out_dir / "cartoon_work.png")

        line_art = {
            "version": "1.0",
            "source_cartoon": str(work_cartoon),
            "line_mask_path": str(line_path),
            "strokes_mask_path": str(strokes_path),
            "flat_path": str(flat_path),
            "subject_mask_path": str(subject_path),
            "work_size": {"width": w, "height": h},
            "methods": methods,
            "difficulty": preset.name,
        }
        return ToolResult(
            success=True,
            data={"line_art": line_art, "line_fraction": round(float(lines[subject].mean()), 4), "warnings": warnings},
            artifacts=[str(line_path), str(strokes_path), str(flat_path), str(subject_path)],
            duration_seconds=time.monotonic() - started,
        )
