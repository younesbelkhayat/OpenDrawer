"""Foreground segmentation: subject mask + cutout on a white background.

Primary path: rembg with BiRefNet / IS-Net (local ONNX models, downloaded once
by rembg into ~/.u2net). Fallback path: OpenCV GrabCut seeded with the subject
boxes from ``subject_detect`` — no model at all, lower quality, always works.

The mask is cleaned for coloring-book use: specks are removed, small holes are
filled, and only components touching a detected subject are kept.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lib.imaging import composite_on_white, disk, load_rgb, save_mask, save_rgb
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

REMBG_MODELS = ("birefnet-portrait", "birefnet-general", "isnet-general-use", "u2net_human_seg", "u2net")


def _rembg_available() -> bool:
    try:
        import rembg  # noqa: F401

        return True
    except Exception:
        return False


def rembg_mask(rgb: np.ndarray, model: str) -> np.ndarray:
    import rembg
    from PIL import Image

    session = rembg.new_session(model)
    out = rembg.remove(Image.fromarray(rgb), session=session, only_mask=True, post_process_mask=True)
    return np.array(out.convert("L")) > 127


def grabcut_mask(rgb: np.ndarray, boxes: list[list[int]] | None, iterations: int = 6) -> np.ndarray:
    """GrabCut seeded with subject boxes (or a centered box when none)."""
    h, w = rgb.shape[:2]
    scale = min(1.0, 800 / max(h, w))
    small = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    gc = np.full((sh, sw), cv2.GC_BGD, np.uint8)
    if not boxes:
        boxes = [[int(w * 0.1), int(h * 0.05), int(w * 0.8), int(h * 0.9)]]
    for x, y, bw, bh in boxes:
        x0, y0 = int(x * scale), int(y * scale)
        x1, y1 = int((x + bw) * scale), int((y + bh) * scale)
        gc[max(0, y0):min(sh, y1), max(0, x0):min(sw, x1)] = cv2.GC_PR_FGD
        # The central part of each box is almost surely the subject.
        cx0, cy0 = x0 + (x1 - x0) // 3, y0 + (y1 - y0) // 4
        cx1, cy1 = x1 - (x1 - x0) // 3, y1 - (y1 - y0) // 4
        if cx1 > cx0 and cy1 > cy0:
            gc[cy0:cy1, cx0:cx1] = cv2.GC_FGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
    fg = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    return cv2.resize(fg, (w, h), interpolation=cv2.INTER_LINEAR) > 0


def clean_mask(
    mask: np.ndarray,
    boxes: list[list[int]] | None = None,
    *,
    min_component_frac: float = 0.004,
    max_hole_frac: float = 0.002,
) -> np.ndarray:
    """Remove specks, fill small holes, keep components that touch a subject box."""
    h, w = mask.shape
    area = h * w
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, disk(max(1, int(min(h, w) * 0.002))))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_component_frac * area:
            continue
        if boxes:
            x, y, bw, bh = stats[i, :4]
            touches = any(
                x < bx + bbw and bx < x + bw and y < by + bbh and by < y + bh for bx, by, bbw, bbh in boxes
            )
            if not touches:
                continue
        keep[i] = True
    out = keep[labels]

    inv = (~out).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    for i in range(1, n):
        x, y, bw, bh, a = stats[i]
        on_border = x == 0 or y == 0 or x + bw == w or y + bh == h
        if not on_border and a < max_hole_frac * area:
            out[labels == i] = True
    return out


class BgRemove(BaseTool):
    name = "bg_remove"
    version = "1.0.0"
    tier = ToolTier.ANALYZE
    capability = "segmentation"
    provider = "rembg"
    stability = ToolStability.BETA
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.HYBRID

    dependencies = ["python:cv2"]
    install_instructions = (
        "pip install 'rembg[gpu]'   # BiRefNet/IS-Net (recommandé, poids téléchargés une fois)\n"
        "Sans rembg, l'outil utilise GrabCut (OpenCV), moins précis."
    )
    capabilities = ["background_removal", "subject_mask", "white_background_cutout"]
    best_for = ["détourer personnes et animaux sur fond blanc"]
    not_good_for = ["sujets transparents ou très flous"]
    required_models = ["rembg_birefnet_portrait", "rembg_isnet_general"]

    input_schema = {
        "type": "object",
        "required": ["input_path", "mask_path", "cutout_path"],
        "properties": {
            "input_path": {"type": "string"},
            "mask_path": {"type": "string", "description": "Output mask PNG (white = subject)"},
            "cutout_path": {"type": "string", "description": "Output RGB cutout on white"},
            "model": {
                "type": "string",
                "enum": ["auto", "grabcut", *REMBG_MODELS],
                "default": "auto",
                "description": "auto = birefnet-portrait for people only, isnet-general-use otherwise",
            },
            "subject_kinds": {"type": "array", "items": {"type": "string"}},
            "subject_boxes": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}},
                "description": "[x, y, w, h] boxes from subject_detect; used to seed GrabCut and drop stray blobs",
            },
            "feather_px": {"type": "integer", "default": 1},
        },
    }
    resource_profile = ResourceProfile(cpu_cores=2, ram_mb=3000, vram_mb=0, disk_mb=1000)
    idempotency_key_fields = ["input_path", "model", "subject_boxes"]
    side_effects = ["writes mask_path and cutout_path"]
    user_visible_verification = ["Cheveux, oreilles, mains et queue conservés ; fond entièrement blanc"]

    def get_status(self) -> ToolStatus:
        try:
            import cv2  # noqa: F401

            return ToolStatus.AVAILABLE
        except ImportError:
            return ToolStatus.UNAVAILABLE

    @staticmethod
    def _pick_model(requested: str, kinds: list[str]) -> str:
        if requested != "auto":
            return requested
        if not _rembg_available():
            return "grabcut"
        if kinds and all(k == "person" for k in kinds):
            return "birefnet-portrait"
        return "isnet-general-use"

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        started = time.monotonic()
        src = Path(inputs["input_path"])
        if not src.exists():
            return ToolResult(success=False, error=f"Input not found: {src}")
        boxes = inputs.get("subject_boxes") or None
        model = self._pick_model(inputs.get("model", "auto"), inputs.get("subject_kinds") or [])
        rgb = load_rgb(src)

        warnings: list[str] = []
        try:
            if model == "grabcut":
                raw = grabcut_mask(rgb, boxes)
            else:
                try:
                    raw = rembg_mask(rgb, model)
                except Exception as exc:
                    warnings.append(f"rembg ({model}) failed, fell back to GrabCut: {exc}")
                    model = "grabcut"
                    raw = grabcut_mask(rgb, boxes)
            mask = clean_mask(raw, boxes)
            if not mask.any():
                warnings.append("Mask empty after cleanup; kept raw mask")
                mask = raw
        except Exception as exc:
            return ToolResult(success=False, error=f"bg_remove failed: {exc}")

        cutout = composite_on_white(rgb, mask, feather_px=int(inputs.get("feather_px", 1)))
        save_mask(mask, inputs["mask_path"])
        save_rgb(cutout, inputs["cutout_path"])
        return ToolResult(
            success=True,
            data={
                "mask_path": str(inputs["mask_path"]),
                "cutout_path": str(inputs["cutout_path"]),
                "model": model,
                "foreground_fraction": round(float(mask.mean()), 4),
                "warnings": warnings,
            },
            artifacts=[str(inputs["mask_path"]), str(inputs["cutout_path"])],
            model=model,
            duration_seconds=time.monotonic() - started,
        )
