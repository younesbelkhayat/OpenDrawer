"""Algorithmic cartoonization (no model, no GPU).

Edge-preserving smoothing + color quantization in LAB + dark outlines from an
adaptive threshold. The result keeps the photo's exact geometry and colors —
less charming than diffusion, but instant and always available. Used as the
fallback backend, for quick previews, and by the test suite.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from lib.imaging import kmeans_lab, lab_to_rgb, resize_long_side, rgb_to_lab, subsample
from tools.base_tool import ResourceProfile, ToolRuntime, ToolStability, ToolStatus
from tools.cartoon._common import Job
from tools.cartoon._provider import CartoonizeProvider


def cartoonize_opencv(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    smoothing_passes: int = 6,
    colors: int = 10,
    edge_block_size: int = 9,
    edge_c: float = 4,
    seed: int = 0,
) -> np.ndarray:
    work = resize_long_side(rgb, min(1280, max(rgb.shape[:2])))
    m = cv2.resize(mask.astype(np.uint8), (work.shape[1], work.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

    smooth = work.copy()
    for _ in range(max(0, smoothing_passes)):
        smooth = cv2.bilateralFilter(smooth, 9, 40, 9)

    lab = rgb_to_lab(smooth)
    fg = lab[m] if m.any() else lab.reshape(-1, 3)
    centers, _ = kmeans_lab(subsample(fg, 30000, seed=seed), colors, seed=seed)
    flat = lab.reshape(-1, 3)
    dist = np.linalg.norm(flat[:, None, :] - centers[None, :, :], axis=2)
    quant = centers[np.argmin(dist, axis=1)].reshape(lab.shape)
    out = lab_to_rgb(quant)
    out = cv2.medianBlur(out, 5)

    gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 7)
    block = edge_block_size | 1
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, edge_c) == 0
    edges = cv2.morphologyEx(edges.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    outline = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    out[(edges & m) | outline] = (25, 25, 25)
    out[~m] = 255
    if out.shape[:2] != rgb.shape[:2]:
        out = cv2.resize(out, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    return out


class OpenCVCartoonize(CartoonizeProvider):
    name = "opencv_cartoonize"
    version = "1.0.0"
    provider = "opencv"
    stability = ToolStability.BETA
    runtime = ToolRuntime.LOCAL
    dependencies = ["python:cv2"]
    install_instructions = "pip install opencv-python-headless"
    capabilities = ["photo_to_cartoon", "instant_preview"]
    best_for = ["aperçu instantané", "machine sans GPU", "tests"]
    not_good_for = ["style livre jeunesse soigné", "ressemblance stylisée"]
    resource_profile = ResourceProfile(cpu_cores=2, ram_mb=800)

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 2.0 * int(inputs.get("candidates", 3))

    def describe(self, ctx: dict[str, Any]) -> dict[str, Any]:
        return {"model_family": None, "identity_mode": "none", "model": "opencv"}

    def generate_job(self, job: Job, crop_rgb: np.ndarray, seed: int, ctx: dict[str, Any]) -> np.ndarray:
        params = dict(ctx["style"].get("opencv") or {})
        # Vary the look slightly between candidates.
        variant = seed % 3
        params["colors"] = int(params.get("colors", 10)) + (-2, 0, 3)[variant]
        params["smoothing_passes"] = int(params.get("smoothing_passes", 6)) + (0, 2, -2)[variant]
        return cartoonize_opencv(crop_rgb, job.mask, seed=seed, **params)
