"""Line extraction + region segmentation on the synthetic character."""

from __future__ import annotations

import numpy as np
import pytest

from lib.difficulty import WORK_LONG_SIDE, get_preset
from lib.imaging import load_mask, load_rgb, resize_long_side, save_mask, save_rgb
from opendrawer.demo import draw_character
from tools.coloring.lineart_extract import LineartExtract
from tools.coloring.region_segment import RegionSegment, decode_labels, encode_labels


@pytest.fixture(scope="module")
def cartoon_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("cartoon")
    img, mask = draw_character()
    return save_rgb(img, d / "cartoon.png"), save_mask(mask, d / "mask.png"), d


def run(cartoon_files, difficulty, tmp_path):
    cartoon, mask, _ = cartoon_files
    la = LineartExtract().execute(
        {"cartoon_path": str(cartoon), "foreground_mask_path": str(mask), "output_dir": str(tmp_path / "la"),
         "difficulty": difficulty, "neural": "off"}
    )
    assert la.success, la.error
    rs = RegionSegment().execute({"line_art": la.data["line_art"], "output_dir": str(tmp_path / "rs")})
    assert rs.success, rs.error
    return la.data["line_art"], rs.data["region_map"]


def test_label_encoding_roundtrip():
    labels = np.array([[0, 1, 300], [70000, 5, 0]], np.int32)
    assert (decode_labels(encode_labels(labels)) == labels).all()


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_zones_are_simplified_and_pupils_black(cartoon_files, difficulty, tmp_path):
    line_art, rm = run(cartoon_files, difficulty, tmp_path)
    preset = get_preset(difficulty)
    assert max(line_art["work_size"].values()) == WORK_LONG_SIDE
    assert 6 <= rm["region_count"] <= preset.max_regions
    # The two pupils are compact dark islands → pre-filled black.
    assert rm["black_fill_count"] == 2
    names = {r["color_name"] for r in rm["regions"] if not r["black_fill"]}
    assert {"bleu", "jaune"} <= names
    labels = decode_labels(load_rgb(rm["labels_path"]))
    assert labels.max() == len(rm["regions"])


def test_harder_means_thinner_lines(cartoon_files, tmp_path):
    widths = {}
    for d in ("easy", "hard"):
        _, rm = run(cartoon_files, d, tmp_path / d)
        widths[d] = (~load_mask(rm["lines_path"])).mean()
    assert widths["easy"] > widths["hard"]


def test_colors_snap_to_photo_palette(cartoon_files, tmp_path):
    cartoon, mask, _ = cartoon_files
    la = LineartExtract().execute(
        {"cartoon_path": str(cartoon), "foreground_mask_path": str(mask), "output_dir": str(tmp_path / "la"), "neural": "off"}
    )
    palette = [{"hex": "#3C6EC8"}, {"hex": "#F5CDAF"}]
    rs = RegionSegment().execute({"line_art": la.data["line_art"], "palette": palette, "output_dir": str(tmp_path / "rs")})
    hexes = {r["color_hex"] for r in rs.data["region_map"]["regions"]}
    assert "#3C6EC8" in hexes and "#F5CDAF" in hexes


def test_neural_on_without_weights_fails_cleanly(cartoon_files, tmp_path):
    cartoon, mask, _ = cartoon_files
    res = LineartExtract().execute(
        {"cartoon_path": str(cartoon), "foreground_mask_path": str(mask), "output_dir": str(tmp_path), "neural": "on"}
    )
    assert not res.success and "neural lineart" in res.error


def test_resize_long_side_keeps_masks_boolean():
    m = np.zeros((10, 20), bool)
    m[2:5, 3:9] = True
    out = resize_long_side(m, 40)
    assert out.dtype == bool and out.shape == (20, 40)
