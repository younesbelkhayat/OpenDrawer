"""Extract the real colors of a photo (skin, hair, clothes, fur).

K-means in CIE LAB over the foreground pixels, then near-duplicate colors are
merged. The resulting palette is what the region stage snaps zone colors to,
so the colored model uses the photo's actual colors.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from lib.imaging import (
    delta_e,
    kmeans_lab,
    lab_to_hex,
    load_mask,
    load_rgb,
    nearest_crayon,
    resize_long_side,
    rgb_to_lab,
    subsample,
)
from tools.base_tool import (
    BaseTool,
    Determinism,
    ResourceProfile,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolTier,
)


def extract_palette(
    rgb: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    n_colors: int = 12,
    merge_delta_e: float = 8.0,
    min_weight: float = 0.008,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Return palette entries sorted by weight (share of foreground pixels)."""
    small = resize_long_side(rgb, 384)
    if mask is not None:
        small_mask = resize_long_side(mask, 384)
        pixels = small[small_mask]
    else:
        pixels = small.reshape(-1, 3)
        # Without a mask, pure white is most likely the cutout background.
        pixels = pixels[~np.all(pixels >= 250, axis=1)]
    if len(pixels) == 0:
        return [{"hex": "#FFFFFF", "lab": [100.0, 0.0, 0.0], "weight": 1.0, "name": "blanc"}]

    lab = rgb_to_lab(subsample(pixels, 40000, seed=seed))
    centers, labels = kmeans_lab(lab, n_colors, seed=seed)
    counts = np.bincount(labels, minlength=len(centers)).astype(np.float64)

    # Greedy merge of near-identical centers (weighted average).
    order = list(np.argsort(-counts))
    merged: list[tuple[np.ndarray, float]] = []
    for idx in order:
        c, w = centers[idx], counts[idx]
        for j, (mc, mw) in enumerate(merged):
            if float(delta_e(mc, c)) < merge_delta_e:
                merged[j] = ((mc * mw + c * w) / (mw + w), mw + w)
                break
        else:
            merged.append((c, w))

    total = sum(w for _, w in merged) or 1.0
    palette = []
    for c, w in sorted(merged, key=lambda item: -item[1]):
        weight = float(w / total)
        if weight < min_weight:
            continue
        name, _ = nearest_crayon(c)
        palette.append(
            {
                "hex": lab_to_hex(c),
                "lab": [round(float(v), 2) for v in c],
                "weight": round(weight, 4),
                "name": name,
            }
        )
    return palette


class PaletteExtract(BaseTool):
    name = "palette_extract"
    version = "1.0.0"
    tier = ToolTier.ANALYZE
    capability = "color_analysis"
    provider = "opencv"
    stability = ToolStability.BETA
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    dependencies = ["python:cv2", "python:numpy"]
    install_instructions = "pip install opencv-python-headless numpy"
    capabilities = ["palette_extraction", "real_photo_colors"]
    best_for = ["couleurs réelles de la photo pour le modèle coloré"]

    input_schema = {
        "type": "object",
        "required": ["image_path"],
        "properties": {
            "image_path": {"type": "string"},
            "mask_path": {"type": "string", "description": "Foreground mask (white = subject)"},
            "n_colors": {"type": "integer", "default": 12, "minimum": 2, "maximum": 40},
            "merge_delta_e": {"type": "number", "default": 8.0},
            "min_weight": {"type": "number", "default": 0.008},
            "seed": {"type": "integer", "default": 0},
        },
    }
    output_schema = {
        "type": "object",
        "properties": {"palette": {"type": "array"}},
    }
    resource_profile = ResourceProfile(cpu_cores=1, ram_mb=400)
    idempotency_key_fields = ["image_path", "mask_path", "n_colors", "merge_delta_e", "seed"]

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        started = time.monotonic()
        try:
            rgb = load_rgb(inputs["image_path"])
            mask = load_mask(inputs["mask_path"]) if inputs.get("mask_path") else None
            if mask is not None and mask.shape != rgb.shape[:2]:
                return ToolResult(success=False, error="mask_path size does not match image_path")
            palette = extract_palette(
                rgb,
                mask,
                n_colors=int(inputs.get("n_colors", 12)),
                merge_delta_e=float(inputs.get("merge_delta_e", 8.0)),
                min_weight=float(inputs.get("min_weight", 0.008)),
                seed=int(inputs.get("seed", 0)),
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"palette_extract failed: {exc}")
        return ToolResult(
            success=True,
            data={"palette": palette},
            duration_seconds=time.monotonic() - started,
        )
