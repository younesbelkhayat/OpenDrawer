"""Compose the printable page: big drawing to color + small colored model.

A4 portrait at 300 dpi by default. The drawing is made as large as possible
while the thumbnail (bottom-right corner) never overlaps its ink: several
placements are tried at decreasing scales, and the best one wins. The same
layout is written as PNG (raster, print-ready) and SVG (vector, built from
the zone SVGs so it stays fillable).
"""

from __future__ import annotations

import copy
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lib.imaging import load_rgb, mask_bbox, save_rgb
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

PAPERS_MM = {"A4": (210.0, 297.0)}
FRAME_COLOR = (150, 150, 150)
SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


def page_size_px(paper: str, dpi: int) -> tuple[int, int]:
    if paper not in PAPERS_MM:
        raise ValueError(f"Unsupported paper {paper!r}; supported: {list(PAPERS_MM)}")
    w_mm, h_mm = PAPERS_MM[paper]
    return round(w_mm / 25.4 * dpi), round(h_mm / 25.4 * dpi)


def ink_crop_box(img: np.ndarray, pad_frac: float = 0.01) -> tuple[int, int, int, int]:
    """Bounding box of the non-white content, padded."""
    ink = np.any(img < 245, axis=2)
    box = mask_bbox(ink)
    h, w = ink.shape
    if box is None:
        return 0, 0, w, h
    x, y, bw, bh = box
    pad = int(max(h, w) * pad_frac)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    return x0, y0, x1 - x0, y1 - y0


def plan_layout(
    ink: np.ndarray,
    content: tuple[int, int, int, int],
    thumb_size: tuple[int, int],
    gap: int,
) -> dict[str, Any]:
    """Largest drawing placement that keeps the thumbnail corner free."""
    cx, cy, cw, ch = content
    dh, dw = ink.shape
    tw, th = thumb_size
    thumb = (cx + cw - tw, cy + ch - th, tw, th)
    s_max = min(cw / dw, ch / dh)

    # Coarse ink mask for fast overlap tests.
    q = max(1, int(max(dh, dw) / 300))
    small = cv2.resize(ink.astype(np.uint8), (max(1, dw // q), max(1, dh // q)), interpolation=cv2.INTER_AREA) > 0

    def overlaps(x: float, y: float, s: float) -> bool:
        tx0, ty0 = thumb[0] - gap, thumb[1] - gap
        # Thumbnail rect in drawing (small-mask) coordinates.
        sx0 = int((tx0 - x) / (s * q))
        sy0 = int((ty0 - y) / (s * q))
        if sx0 >= small.shape[1] or sy0 >= small.shape[0]:
            return False
        return bool(small[max(0, sy0):, max(0, sx0):].any())

    for s in np.linspace(s_max, s_max * 0.45, 28):
        w, h = dw * s, dh * s
        candidates = [
            (cx + (cw - w) / 2, cy),  # centered, top
            (cx, cy),  # left, top
            (cx + (cw - w) / 2, cy + (ch - h) / 2),  # centered
        ]
        for x, y in candidates:
            if overlaps(x, y, s):
                continue
            return {"scale": float(s), "x": float(x), "y": float(y), "thumb": thumb, "mode": "overlay"}
    # Fallback: reserve a bottom band for the thumbnail.
    band_h = ch - th - gap
    s = min(cw / dw, band_h / dh)
    return {
        "scale": float(s),
        "x": float(cx + (cw - dw * s) / 2),
        "y": float(cy),
        "thumb": thumb,
        "mode": "band",
    }


def compose_png(
    page_w: int,
    page_h: int,
    drawing: np.ndarray,
    model: np.ndarray,
    layout: dict[str, Any],
    frame_px: int,
    pad_px: int,
) -> np.ndarray:
    page = np.full((page_h, page_w, 3), 255, np.uint8)
    s = layout["scale"]
    dw, dh = max(1, round(drawing.shape[1] * s)), max(1, round(drawing.shape[0] * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    d = cv2.resize(drawing, (dw, dh), interpolation=interp)
    x, y = round(layout["x"]), round(layout["y"])
    x1, y1 = min(page_w, x + dw), min(page_h, y + dh)
    region = page[y:y1, x:x1]
    page[y:y1, x:x1] = np.minimum(region, d[: y1 - y, : x1 - x])

    tx, ty, tw, th = layout["thumb"]
    inner_w, inner_h = tw - 2 * pad_px, th - 2 * pad_px
    ms = min(inner_w / model.shape[1], inner_h / model.shape[0])
    mw, mh = max(1, round(model.shape[1] * ms)), max(1, round(model.shape[0] * ms))
    m = cv2.resize(model, (mw, mh), interpolation=cv2.INTER_AREA)
    mx, my = tx + (tw - mw) // 2, ty + (th - mh) // 2
    page[my:my + mh, mx:mx + mw] = m
    radius = max(4, pad_px)
    page_bgr = np.ascontiguousarray(page)
    _rounded_rect(page_bgr, (tx, ty), (tx + tw, ty + th), FRAME_COLOR, frame_px, radius)
    return page_bgr


def _rounded_rect(img: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], color, thickness: int, r: int) -> None:
    x0, y0 = p0
    x1, y1 = p1
    lt = cv2.LINE_AA
    cv2.line(img, (x0 + r, y0), (x1 - r, y0), color, thickness, lt)
    cv2.line(img, (x0 + r, y1), (x1 - r, y1), color, thickness, lt)
    cv2.line(img, (x0, y0 + r), (x0, y1 - r), color, thickness, lt)
    cv2.line(img, (x1, y0 + r), (x1, y1 - r), color, thickness, lt)
    cv2.ellipse(img, (x0 + r, y0 + r), (r, r), 180, 0, 90, color, thickness, lt)
    cv2.ellipse(img, (x1 - r, y0 + r), (r, r), 270, 0, 90, color, thickness, lt)
    cv2.ellipse(img, (x0 + r, y1 - r), (r, r), 90, 0, 90, color, thickness, lt)
    cv2.ellipse(img, (x1 - r, y1 - r), (r, r), 0, 0, 90, color, thickness, lt)


def _svg_children(path: str, id_prefix: str = "") -> list[ET.Element]:
    root = ET.parse(path).getroot()
    out = []
    for child in root:
        if child.tag.endswith("title"):
            continue
        child = copy.deepcopy(child)
        if id_prefix:
            for el in child.iter():
                if "id" in el.attrib:
                    el.attrib["id"] = id_prefix + el.attrib["id"]
        out.append(child)
    return out


def compose_svg(
    page_w: int,
    page_h: int,
    paper_mm: tuple[float, float],
    coloring_svg: str,
    model_svg: str,
    crop: tuple[int, int, int, int],
    layout: dict[str, Any],
    model_scale: float,
    model_offset: tuple[float, float],
    frame_px: int,
    pad_px: int,
) -> str:
    svg = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "viewBox": f"0 0 {page_w} {page_h}",
            "width": f"{paper_mm[0]:g}mm",
            "height": f"{paper_mm[1]:g}mm",
            "data-generator": "OpenDrawer",
        },
    )
    ET.SubElement(svg, f"{{{SVG_NS}}}rect", {"width": str(page_w), "height": str(page_h), "fill": "#FFFFFF"})
    s = layout["scale"]
    cx0, cy0 = crop[0], crop[1]
    drawing = ET.SubElement(
        svg,
        f"{{{SVG_NS}}}g",
        {"id": "coloriage", "transform": f"translate({layout['x'] - cx0 * s:.2f} {layout['y'] - cy0 * s:.2f}) scale({s:.5f})"},
    )
    for child in _svg_children(coloring_svg):
        drawing.append(child)

    tx, ty, tw, th = layout["thumb"]
    ms = model_scale
    mx, my = model_offset
    thumb = ET.SubElement(
        svg,
        f"{{{SVG_NS}}}g",
        {"id": "modele", "transform": f"translate({mx - cx0 * ms:.2f} {my - cy0 * ms:.2f}) scale({ms:.5f})"},
    )
    for child in _svg_children(model_svg, "modele-"):
        thumb.append(child)
    ET.SubElement(
        svg,
        f"{{{SVG_NS}}}rect",
        {
            "id": "cadre-modele",
            "x": str(tx),
            "y": str(ty),
            "width": str(tw),
            "height": str(th),
            "rx": str(max(4, pad_px)),
            "fill": "none",
            "stroke": "#969696",
            "stroke-width": str(frame_px),
        },
    )
    return ET.tostring(svg, encoding="unicode", xml_declaration=False) + "\n"


class PageLayout(BaseTool):
    name = "page_layout"
    version = "1.0.0"
    tier = ToolTier.LAYOUT
    capability = "page_layout"
    provider = "opendrawer"
    stability = ToolStability.BETA
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    dependencies = ["python:cv2"]
    install_instructions = "pip install opencv-python-headless"
    capabilities = ["a4_page", "thumbnail_model", "png_export", "svg_export"]
    best_for = ["page imprimable avec modèle colorié en vignette"]

    input_schema = {
        "type": "object",
        "required": ["region_map", "coloring_svg", "model_svg", "output_dir"],
        "properties": {
            "region_map": {"type": "object"},
            "coloring_svg": {"type": "string"},
            "model_svg": {"type": "string"},
            "output_dir": {"type": "string"},
            "paper": {"type": "string", "enum": list(PAPERS_MM), "default": "A4"},
            "dpi": {"type": "integer", "default": 300},
            "margin_mm": {"type": "number", "default": 10.0},
            "thumbnail_ratio": {"type": "number", "default": 0.26, "description": "Thumbnail width / content width"},
        },
    }
    resource_profile = ResourceProfile(cpu_cores=1, ram_mb=1500)

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        started = time.monotonic()
        try:
            rm = inputs["region_map"]
            paper = inputs.get("paper", "A4")
            dpi = int(inputs.get("dpi", 300))
            page_w, page_h = page_size_px(paper, dpi)
            coloring_work = load_rgb(rm["coloring_png"])
            model_work = load_rgb(rm["model_png"])
        except Exception as exc:
            return ToolResult(success=False, error=f"page_layout: invalid inputs: {exc}")

        crop = ink_crop_box(coloring_work)
        x0, y0, cw_, ch_ = crop
        drawing = coloring_work[y0:y0 + ch_, x0:x0 + cw_]
        model = model_work[y0:y0 + ch_, x0:x0 + cw_]

        margin = round(float(inputs.get("margin_mm", 10.0)) / 25.4 * dpi)
        content = (margin, margin, page_w - 2 * margin, page_h - 2 * margin)
        frame_px = max(2, round(dpi / 100))
        pad_px = max(6, round(dpi / 25))
        gap = max(10, round(dpi / 12))
        tw = round(content[2] * float(inputs.get("thumbnail_ratio", 0.26)))
        th = round((tw - 2 * pad_px) * model.shape[0] / model.shape[1]) + 2 * pad_px
        th = min(th, round(content[3] * 0.4))

        ink = np.any(drawing < 245, axis=2)
        layout = plan_layout(ink, content, (tw, th), gap)
        page_png = compose_png(page_w, page_h, drawing, model, layout, frame_px, pad_px)

        tx, ty, tw_, th_ = layout["thumb"]
        inner_w, inner_h = tw_ - 2 * pad_px, th_ - 2 * pad_px
        ms = min(inner_w / model.shape[1], inner_h / model.shape[0])
        mw, mh = model.shape[1] * ms, model.shape[0] * ms
        model_offset = (tx + (tw_ - mw) / 2, ty + (th_ - mh) / 2)

        out = Path(inputs["output_dir"])
        try:
            page_svg = compose_svg(
                page_w,
                page_h,
                PAPERS_MM[paper],
                inputs["coloring_svg"],
                inputs["model_svg"],
                crop,
                layout,
                ms,
                model_offset,
                frame_px,
                pad_px,
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"page_layout: could not build page SVG: {exc}")

        page_png_path = save_rgb(page_png, out / "page.png")
        page_svg_path = out / "page.svg"
        page_svg_path.write_text(page_svg, encoding="utf-8")
        coloring_png_path = save_rgb(drawing, out / "coloriage.png")
        model_png_path = save_rgb(model, out / "modele.png")

        s = layout["scale"]
        drawing_box = [round(layout["x"]), round(layout["y"]), round(cw_ * s), round(ch_ * s)]
        coloring_page = {
            "version": "1.0",
            "paper": paper,
            "dpi": dpi,
            "page_size": {"width": page_w, "height": page_h},
            "page_png": str(page_png_path),
            "page_svg": str(page_svg_path),
            "coloring_png": str(coloring_png_path),
            "coloring_svg": str(inputs["coloring_svg"]),
            "model_png": str(model_png_path),
            "model_svg": str(inputs["model_svg"]),
            "drawing_box": drawing_box,
            "thumbnail_box": [int(v) for v in layout["thumb"]],
            "crop_box": [int(v) for v in crop],
            "layout_mode": layout["mode"],
        }
        return ToolResult(
            success=True,
            data={"coloring_page": coloring_page},
            artifacts=[str(page_png_path), str(page_svg_path), str(coloring_png_path), str(model_png_path)],
            duration_seconds=time.monotonic() - started,
        )

