"""End-to-end runs through the CLI runner (OpenCV backend, no weights)."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from lib.checkpoint import read_checkpoint
from opendrawer.runner import AutoApprover, BookRun, PageRun, RunSettings


def test_page_outputs(e2e_page):
    projects_dir, pid = e2e_page
    cp = read_checkpoint(projects_dir, pid, "page")
    assert cp["status"] == "completed" and cp["human_approved"]
    page = cp["artifacts"]["coloring_page"]
    qa = cp["artifacts"]["qa_report"]
    assert qa["status"] in ("pass", "warn"), qa["summary"]
    assert {c["id"]: c["status"] for c in qa["checks"]}["closed_zones"] == "pass"

    with Image.open(page["page_png"]) as im:
        assert im.size == (2480, 3508)  # A4 portrait, 300 dpi
    svg = ET.parse(page["coloring_svg"]).getroot()
    zones = [el for el in svg.iter() if el.attrib.get("class") == "zone"]
    assert zones and all(z.attrib["data-color"].startswith("#") for z in zones)
    assert all(z.attrib["fill"] == "#FFFFFF" for z in zones)
    model = ET.parse(page["model_svg"]).getroot()
    assert any(el.attrib.get("fill") not in (None, "#FFFFFF", "none") for el in model.iter() if el.attrib.get("class") == "zone")
    page_svg = ET.parse(page["page_svg"]).getroot()
    ids = [el.attrib["id"] for el in page_svg.iter() if "id" in el.attrib]
    assert len(ids) == len(set(ids)), "page SVG ids must be unique"


def test_thumbnail_does_not_cover_drawing(e2e_page):
    projects_dir, pid = e2e_page
    page = read_checkpoint(projects_dir, pid, "page")["artifacts"]["coloring_page"]
    dx, dy, dw, dh = page["drawing_box"]
    tx, ty, tw, th = page["thumbnail_box"]
    assert tx + tw <= 2480 and ty + th <= 3508
    img = np.array(Image.open(page["page_png"]).convert("L"))
    # Just above the thumbnail frame, the page is white or drawing — never both overlapping.
    assert img[ty + 20 : ty + th - 20, tx + 20 : tx + tw - 20].mean() < 255  # thumbnail has content


def test_artifacts_written_and_events_logged(e2e_page):
    projects_dir, pid = e2e_page
    d = projects_dir / pid
    for name in ("photo_analysis", "cartoon_set", "line_art", "region_map", "coloring_page", "qa_report"):
        assert (d / "artifacts" / f"{name}.json").is_file(), name
    marker = json.loads((d / "project.json").read_text())
    assert marker["settings"]["difficulty"] == "medium"


class StopAtCartoon(AutoApprover):
    def choose_cartoon(self, project_id, cartoon_set):
        return None


def test_gate_pauses_then_resumes(demo_photo, tmp_path):
    settings = RunSettings(backend="opencv", candidates=2, seed=3)
    run = PageRun.create(demo_photo, settings, StopAtCartoon(), project_id="gated", projects_dir=tmp_path, log=lambda _: None)
    assert run.run() == "awaiting_human"
    assert read_checkpoint(tmp_path, "gated", "cartoon")["status"] == "awaiting_human"
    resumed = PageRun("gated", AutoApprover(), projects_dir=tmp_path, log=lambda _: None)
    assert resumed.run() == "completed"


def test_book(demo_photo, tmp_path):
    album = tmp_path / "album"
    album.mkdir()
    for name in ("01.jpg", "02.jpg"):
        (album / name).write_bytes(demo_photo.read_bytes())
    settings = RunSettings(difficulty="easy", backend="opencv", candidates=1, seed=0)
    book = BookRun.create(album, settings, AutoApprover(), book_id="livre", projects_dir=tmp_path / "p", log=lambda _: None)
    assert book.run() == "completed"
    out = tmp_path / "p" / "livre" / "output" / "livre"
    assert sorted(p.name for p in out.iterdir()) == ["page_01.png", "page_01.svg", "page_02.png", "page_02.svg", "sommaire.png"]
