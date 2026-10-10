"""Difficulty presets for coloring pages.

One knob for the user ("facile / moyen / difficile"), many derived numbers for
the tools. Every length is expressed in pixels at the *work resolution*
(``WORK_LONG_SIDE``): region segmentation runs at that size, and the page
layout only ever downsamples from it, which keeps lines anti-aliased.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

# Long side, in pixels, of the image the region stage works on. An A4 page at
# 300 dpi has a ~2240 x 3270 px drawing box, so the drawing is always scaled
# down (never up) when placed on the page.
WORK_LONG_SIDE = 2400

DIFFICULTY_ALIASES = {
    "facile": "easy",
    "moyen": "medium",
    "difficile": "hard",
}


@dataclass(frozen=True)
class DifficultyPreset:
    name: str
    label_fr: str
    age_hint: str
    # Final stroke widths (work-resolution pixels).
    line_px: int
    outline_px: int
    detail_px: int
    # Region simplification.
    min_region_frac: float  # of the subject area; smaller zones are merged or blackened
    dark_fill_max_frac: float  # compact *dark* zones up to this size are pre-filled black
    min_regions: int
    max_regions: int
    # Line extraction.
    quant_colors: int  # k-means colors used to find color boundaries
    gap_close_px: int  # morphological closing to seal small gaps in lines
    keep_detail_strokes: bool  # keep open strokes (smile, folds) that do not close a zone
    min_detail_len_px: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


PRESETS: dict[str, DifficultyPreset] = {
    "easy": DifficultyPreset(
        name="easy",
        label_fr="facile",
        age_hint="3-5 ans",
        line_px=14,
        outline_px=20,
        detail_px=10,
        min_region_frac=0.006,
        dark_fill_max_frac=0.006,
        min_regions=8,
        max_regions=30,
        quant_colors=8,
        gap_close_px=9,
        keep_detail_strokes=True,
        min_detail_len_px=120,
    ),
    "medium": DifficultyPreset(
        name="medium",
        label_fr="moyen",
        age_hint="6-8 ans",
        line_px=9,
        outline_px=13,
        detail_px=7,
        min_region_frac=0.002,
        dark_fill_max_frac=0.004,
        min_regions=20,
        max_regions=70,
        quant_colors=14,
        gap_close_px=7,
        keep_detail_strokes=True,
        min_detail_len_px=70,
    ),
    "hard": DifficultyPreset(
        name="hard",
        label_fr="difficile",
        age_hint="9-12 ans",
        line_px=6,
        outline_px=9,
        detail_px=5,
        min_region_frac=0.0007,
        dark_fill_max_frac=0.0025,
        min_regions=45,
        max_regions=160,
        quant_colors=22,
        gap_close_px=5,
        keep_detail_strokes=True,
        min_detail_len_px=40,
    ),
}


def get_preset(name: str | None) -> DifficultyPreset:
    """Look up a preset by English or French name (default: medium)."""
    key = (name or "medium").strip().lower()
    key = DIFFICULTY_ALIASES.get(key, key)
    if key not in PRESETS:
        raise ValueError(
            f"Unknown difficulty {name!r}. Use one of: easy/facile, medium/moyen, hard/difficile."
        )
    return PRESETS[key]
