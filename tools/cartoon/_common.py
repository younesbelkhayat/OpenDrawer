"""Shared helpers for the cartoonize tools.

* ``plan_jobs`` splits a photo into generation jobs (whole cutout, or one crop
  per subject when several people/animals each deserve their own face
  conditioning).
* ``compose_jobs`` pastes the per-job results back on a white canvas aligned
  with the original photo — downstream stages rely on that alignment to read
  real colors.
* ``score_candidate`` gives the CLI (and the agent, as a hint) an objective
  ranking: structure kept, colors kept, and how flat/"cartoon" the result is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from lib.imaging import composite_on_white, delta_e, load_mask, load_rgb, rgb_to_lab


@dataclass
class Job:
    """One generation unit: a crop of the cutout plus what is inside it."""

    box: tuple[int, int, int, int]  # x, y, w, h in photo coordinates
    mask: np.ndarray  # bool, crop-sized
    kind: str
    description: str
    face_box: Optional[tuple[int, int, int, int]]  # in crop coordinates
    subject_ids: list[str]


def _expand_box(box: list[int], pad: float, width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = box
    px, py = int(w * pad), int(h * pad)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(width, x + w + px), min(height, y + h + py)
    return x0, y0, x1 - x0, y1 - y0


def load_analysis_images(analysis: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Cutout RGB (white background) and foreground mask from a photo_analysis."""
    cutout = load_rgb(analysis["cutout_path"])
    mask = load_mask(analysis["foreground_mask_path"])
    if mask.shape != cutout.shape[:2]:
        raise ValueError("cutout and foreground mask sizes differ")
    return cutout, mask


def plan_jobs(analysis: dict[str, Any], fg_mask: np.ndarray, per_subject: Optional[bool] = None) -> list[Job]:
    """Whole image for one subject; one crop per subject when several have masks."""
    h, w = fg_mask.shape
    subjects = analysis.get("subjects") or []
    with_masks = [s for s in subjects if s.get("mask_path") and Path(s["mask_path"]).exists()]
    if per_subject is None:
        per_subject = len(with_masks) > 1

    if not per_subject or len(with_masks) < 2:
        kinds = {s.get("kind", "person") for s in subjects} or {"person"}
        kind = kinds.pop() if len(kinds) == 1 else "group"
        faces = [s["face_bbox"] for s in subjects if s.get("face_bbox")]
        face = tuple(faces[0]) if len(faces) == 1 else None
        description = (
            subjects[0].get("description", "a smiling person")
            if len(subjects) == 1
            else _group_description(subjects)
        )
        return [Job((0, 0, w, h), fg_mask.copy(), kind, description, face, [s["id"] for s in subjects])]

    jobs = []
    for subject in with_masks:
        x, y, bw, bh = _expand_box(subject["bbox"], 0.12, w, h)
        sub_mask = load_mask(subject["mask_path"])[y:y + bh, x:x + bw]
        face = None
        if subject.get("face_bbox"):
            fx, fy, fw, fh = subject["face_bbox"]
            face = (fx - x, fy - y, fw, fh)
        jobs.append(
            Job(
                (x, y, bw, bh),
                sub_mask,
                subject.get("kind", "person"),
                subject.get("description", "a smiling person"),
                face,
                [subject["id"]],
            )
        )
    # Paint large subjects first so smaller (usually closer) ones stay on top.
    jobs.sort(key=lambda j: -(j.box[2] * j.box[3]))
    return jobs


def _group_description(subjects: list[dict[str, Any]]) -> str:
    people = sum(1 for s in subjects if s.get("kind") == "person")
    animals = [s.get("label", "animal") for s in subjects if s.get("kind") == "animal"]
    parts = []
    if people:
        parts.append("a smiling person" if people == 1 else f"{people} smiling people")
    for label in animals:
        parts.append(f"a cute {label}")
    return " and ".join(parts) or "a group of friends"


def crop(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    return img[y:y + h, x:x + w]


def compose_jobs(
    size: tuple[int, int],
    results: list[tuple[Job, np.ndarray]],
    feather_px: int = 2,
) -> np.ndarray:
    """Paste each job's (crop-sized) RGB result through its mask on white."""
    h, w = size
    canvas = np.full((h, w, 3), 255, np.uint8)
    for job, rgb in results:
        x, y, bw, bh = job.box
        if rgb.shape[:2] != (bh, bw):
            rgb = cv2.resize(rgb, (bw, bh), interpolation=cv2.INTER_LANCZOS4)
        region = canvas[y:y + bh, x:x + bw]
        alpha = job.mask.astype(np.float32)
        if feather_px:
            k = feather_px * 2 + 1
            alpha = cv2.GaussianBlur(alpha, (k, k), 0)
        alpha = alpha[..., None]
        region[:] = np.clip(rgb * alpha + region * (1 - alpha) + 0.5, 0, 255).astype(np.uint8)
    return canvas


def canny_control(rgb: np.ndarray, low: int = 80, high: int = 180) -> np.ndarray:
    """3-channel Canny edge map (white edges on black), as ControlNet expects."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    edges = cv2.Canny(gray, low, high)
    return np.repeat(edges[..., None], 3, axis=2)


def fit_size(width: int, height: int, long_side: int, multiple: int = 8) -> tuple[int, int]:
    """Size with the given long side, both dimensions a multiple of ``multiple``."""
    scale = long_side / max(width, height)
    return (
        max(multiple, int(round(width * scale / multiple)) * multiple),
        max(multiple, int(round(height * scale / multiple)) * multiple),
    )


def score_candidate(candidate: np.ndarray, photo: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Objective scores in [0, 1] (1 = best)."""
    if candidate.shape != photo.shape:
        candidate = cv2.resize(candidate, (photo.shape[1], photo.shape[0]), interpolation=cv2.INTER_AREA)
    scale = min(1.0, 768 / max(photo.shape[:2]))
    if scale < 1:
        size = (int(photo.shape[1] * scale), int(photo.shape[0] * scale))
        candidate = cv2.resize(candidate, size, interpolation=cv2.INTER_AREA)
        photo = cv2.resize(photo, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    if not mask.any():
        return {"structure": 0.0, "color": 0.0, "style": 0.0, "total": 0.0}

    # Structure: candidate edges should sit on real edges; silhouette should match.
    photo_edges = canny_control(photo)[..., 0] > 0
    cand_edges = canny_control(candidate)[..., 0] > 0
    near_photo = cv2.dilate(photo_edges.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    inside = mask & cand_edges
    precision = float((inside & near_photo).sum() / max(1, inside.sum()))
    cand_fg = ~np.all(candidate >= 245, axis=2)
    iou = float((cand_fg & mask).sum() / max(1, (cand_fg | mask).sum()))
    structure = 0.5 * precision + 0.5 * iou

    # Color: blurred colors should stay close to the photo's.
    blur_c = cv2.GaussianBlur(candidate, (0, 0), 6)
    blur_p = cv2.GaussianBlur(photo, (0, 0), 6)
    de = delta_e(rgb_to_lab(blur_c[mask]), rgb_to_lab(blur_p[mask]))
    color = float(np.clip(1.0 - float(np.median(de)) / 40.0, 0.0, 1.0))

    # Style: share of near-flat pixels (low local variance) inside the subject.
    gray = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (5, 5))
    var = cv2.blur(gray * gray, (5, 5)) - mean * mean
    flat = var < 40.0
    style = float(flat[mask].mean())

    total = 0.4 * structure + 0.35 * color + 0.25 * style
    return {k: round(v, 4) for k, v in {"structure": structure, "color": color, "style": style, "total": total}.items()}


def white_outside(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return composite_on_white(rgb, mask, feather_px=1)
