"""Zero-model demo: draw a synthetic "photo" and run the full pipeline on it.

Used by ``python -m opendrawer demo`` (check an install in seconds, no GPU,
no weights) and by the test suite as a deterministic end-to-end fixture.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def draw_character(width: int = 900, height: int = 1200) -> tuple[np.ndarray, np.ndarray]:
    """A flat cartoon kid (RGB) and its foreground mask (bool)."""
    img = np.full((height, width, 3), 255, np.uint8)
    sx, sy = width / 900, height / 1200
    ink = (25, 25, 25)
    skin, shirt, hair, star_c = (245, 205, 175), (60, 110, 200), (120, 70, 30), (255, 215, 40)

    def P(x, y):
        return int(x * sx), int(y * sy)

    def ell(c, ax, col, t=-1):
        cv2.ellipse(img, P(*c), (int(ax[0] * sx), int(ax[1] * sy)), 0, 0, 360, col, t, cv2.LINE_AA)

    def rect(a, b, col):
        cv2.rectangle(img, P(*a), P(*b), col, -1)
        cv2.rectangle(img, P(*a), P(*b), ink, 6)

    rect((250, 650), (650, 1100), shirt)
    rect((160, 680), (250, 1000), shirt)
    rect((650, 680), (740, 1000), shirt)
    for x in (205, 695):
        ell((x, 1030), (45, 40), skin)
        ell((x, 1030), (45, 40), ink, 6)
    ell((450, 360), (250, 260), hair)
    ell((450, 360), (250, 260), ink, 6)
    ell((450, 420), (200, 220), skin)
    ell((450, 420), (200, 220), ink, 6)
    for x in (380, 520):
        ell((x, 400), (40, 30), (255, 255, 255))
        ell((x, 400), (40, 30), ink, 5)
        ell((x, 400), (14, 14), ink)
    cv2.ellipse(img, P(450, 520), (int(70 * sx), int(40 * sy)), 0, 20, 160, ink, 6, cv2.LINE_AA)
    for x in (440, 462):
        cv2.circle(img, P(x, 465), 5, ink, -1)
    fringe = np.array([P(*p) for p in [(260, 330), (330, 230), (450, 200), (580, 230), (640, 330), (560, 280), (450, 270), (340, 290)]], np.int32)
    cv2.fillPoly(img, [fringe], hair)
    cv2.polylines(img, [fringe], True, ink, 6, cv2.LINE_AA)
    star = np.array([P(*p) for p in [(450, 760), (470, 820), (535, 820), (482, 858), (502, 920), (450, 882), (398, 920), (418, 858), (365, 820), (430, 820)]], np.int32)
    cv2.fillPoly(img, [star], star_c)
    cv2.polylines(img, [star], True, ink, 5, cv2.LINE_AA)
    mask = np.any(img < 245, axis=2).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
    return img, mask


def make_demo_photo(path: str | Path, *, seed: int = 0, width: int = 900, height: int = 1200) -> Path:
    """Write a photo-like JPEG (blur, noise, gradient background) of the demo character."""
    img, mask = draw_character(width, height)
    rng = np.random.default_rng(seed)
    photo = cv2.GaussianBlur(img, (0, 0), 3).astype(np.float32)
    bg = np.linspace(170, 220, width)[None, :, None].repeat(height, 0).repeat(3, 2)
    photo = np.where(mask[..., None], photo, bg) + rng.normal(0, 6, photo.shape)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(photo, 0, 255).astype(np.uint8)).save(path, quality=92)
    return path
