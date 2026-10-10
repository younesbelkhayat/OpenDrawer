"""Small image helpers shared by the coloring tools (numpy + OpenCV + Pillow).

Conventions used everywhere in OpenDrawer:
* RGB images are ``uint8`` arrays of shape (H, W, 3) — never BGR outside cv2 calls.
* Masks are ``bool`` arrays of shape (H, W); on disk they are 8-bit PNGs (0/255).
* LAB values use the CIE scale: L in [0, 100], a/b roughly in [-128, 127].
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps

# Crayon / felt-tip colors a child is likely to own, with French names.
# Used to name the expected color of each zone (SVG data-color-name).
CRAYONS_FR: list[tuple[str, str]] = [
    ("blanc", "#FFFFFF"),
    ("crème", "#FFF3C4"),
    ("jaune", "#FFD92E"),
    ("orange", "#FF8C1A"),
    ("pêche", "#FFC8A2"),
    ("rose pâle", "#F8C8D8"),
    ("rose", "#FF6FA8"),
    ("rouge", "#E3262F"),
    ("bordeaux", "#8C1C2C"),
    ("lilas", "#C8A2D8"),
    ("violet", "#7B3FB3"),
    ("bleu ciel", "#8DCFF0"),
    ("bleu", "#2A6BD1"),
    ("bleu marine", "#1D2A5C"),
    ("turquoise", "#2EC4C4"),
    ("vert clair", "#9BD34A"),
    ("vert", "#2E9443"),
    ("vert foncé", "#1F5A32"),
    ("beige", "#E8C9A0"),
    ("chair claire", "#F6D5B8"),
    ("caramel", "#C68A52"),
    ("marron", "#8B5A2B"),
    ("marron foncé", "#4A2C1A"),
    ("gris clair", "#C9C9C9"),
    ("gris", "#7F7F7F"),
    ("noir", "#1A1A1A"),
]


# ---------------------------------------------------------------- I/O

def load_rgb(path: str | Path, background: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Load any image as RGB uint8, honoring EXIF rotation; alpha → background."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, background + (255,))
            im = Image.alpha_composite(bg, im)
        return np.array(im.convert("RGB"), dtype=np.uint8)


def load_alpha(path: str | Path) -> np.ndarray | None:
    """Alpha channel of an image as a bool mask, or None if it has none."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode not in ("RGBA", "LA"):
            return None
        return np.array(im.getchannel("A")) > 127


def save_rgb(img: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(img, dtype=np.uint8)).save(path)
    return path


def load_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("L")) > 127


def save_mask(mask: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)
    return path


# ---------------------------------------------------------------- geometry

def resize_long_side(img: np.ndarray, long_side: int, interpolation: int | None = None) -> np.ndarray:
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    if abs(scale - 1.0) < 1e-6:
        return img.copy()
    if interpolation is None:
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    if img.dtype == bool:
        return cv2.resize(img.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    return cv2.resize(img, size, interpolation=interpolation)


def resize_to(img: np.ndarray, width: int, height: int, interpolation: int = cv2.INTER_LANCZOS4) -> np.ndarray:
    if img.dtype == bool:
        return cv2.resize(img.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    return cv2.resize(img, (width, height), interpolation=interpolation)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """[x, y, w, h] of the True pixels, or None for an empty mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def disk(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def composite_on_white(rgb: np.ndarray, mask: np.ndarray, feather_px: int = 0) -> np.ndarray:
    """Keep ``mask`` pixels of ``rgb``, paint the rest white."""
    alpha = mask.astype(np.float32)
    if feather_px > 0:
        k = feather_px * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    alpha = alpha[..., None]
    out = rgb.astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- color

def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 (..., 3) → float32 CIE LAB (..., 3)."""
    arr = np.asarray(rgb, dtype=np.uint8)
    shape = arr.shape
    flat = arr.reshape(-1, 1, 3).astype(np.float32) / 255.0
    lab = cv2.cvtColor(flat, cv2.COLOR_RGB2LAB)
    return lab.reshape(shape).astype(np.float32)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    arr = np.asarray(lab, dtype=np.float32)
    shape = arr.shape
    rgb = cv2.cvtColor(arr.reshape(-1, 1, 3), cv2.COLOR_LAB2RGB)
    return np.clip(rgb.reshape(shape) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def rgb_to_hex(rgb: Sequence[int]) -> str:
    r, g, b = (int(v) for v in rgb[:3])
    return f"#{r:02X}{g:02X}{b:02X}"


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def lab_to_hex(lab: Sequence[float]) -> str:
    return rgb_to_hex(lab_to_rgb(np.array(lab, dtype=np.float32)))


def delta_e(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIE76 color difference (Euclidean distance in LAB), broadcasting."""
    return np.linalg.norm(np.asarray(lab1, np.float32) - np.asarray(lab2, np.float32), axis=-1)


_CRAYON_LAB: np.ndarray | None = None


def nearest_crayon(lab: Sequence[float]) -> tuple[str, str]:
    """French name + hex of the closest crayon color."""
    global _CRAYON_LAB
    if _CRAYON_LAB is None:
        _CRAYON_LAB = rgb_to_lab(np.array([hex_to_rgb(h) for _, h in CRAYONS_FR], dtype=np.uint8))
    d = delta_e(_CRAYON_LAB, np.asarray(lab, np.float32)[None, :])
    name, hex_value = CRAYONS_FR[int(np.argmin(d))]
    return name, hex_value


def kmeans_lab(lab_pixels: np.ndarray, k: int, *, seed: int = 0, attempts: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """K-means on (N, 3) LAB pixels → (centers (k, 3), labels (N,))."""
    data = np.ascontiguousarray(lab_pixels, dtype=np.float32)
    k = max(1, min(k, len(data)))
    cv2.setRNGSeed(seed)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    return centers.astype(np.float32), labels.reshape(-1)


def subsample(pixels: np.ndarray, max_count: int, seed: int = 0) -> np.ndarray:
    if len(pixels) <= max_count:
        return pixels
    rng = np.random.default_rng(seed)
    return pixels[rng.choice(len(pixels), size=max_count, replace=False)]


def unique_paths(paths: Iterable[str | Path]) -> list[str]:
    seen: list[str] = []
    for p in paths:
        s = str(p)
        if s not in seen:
            seen.append(s)
    return seen
