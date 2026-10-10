"""Vectorize a region map into fillable SVGs (coloring + colored model).

Each zone becomes one ``<path class="zone">`` (holes via even-odd) carrying
its expected color as ``data-color`` / ``data-color-name``: a coloring app
can fill zones directly, or check a child's choices. Black pre-filled
details and the line art are separate groups drawn on top, so small
smoothing differences between neighboring zones are hidden by the strokes.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr

import cv2
import numpy as np
from scipy import ndimage

from lib.imaging import load_mask, load_rgb
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
from tools.coloring.region_segment import BLACK_HEX, decode_labels

SVG_NS = "http://www.w3.org/2000/svg"


def _fmt(v: float) -> str:
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


def smooth_path(points: np.ndarray) -> str:
    """Closed polygon → smooth path (quadratic curves through edge midpoints)."""
    n = len(points)
    if n < 3:
        return ""
    pts = points.astype(np.float64)
    mids = (pts + np.roll(pts, -1, axis=0)) / 2.0
    parts = [f"M{_fmt(mids[-1][0])} {_fmt(mids[-1][1])}"]
    for i in range(n):
        parts.append(f"Q{_fmt(pts[i][0])} {_fmt(pts[i][1])} {_fmt(mids[i][0])} {_fmt(mids[i][1])}")
    parts.append("Z")
    return "".join(parts)


def mask_to_path(mask: np.ndarray, offset: tuple[int, int] = (0, 0), epsilon: float = 0.9) -> str:
    """Bool mask → path data with holes (use fill-rule="evenodd")."""
    m = np.pad(mask.astype(np.uint8), 1)
    contours, _ = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    ox, oy = offset[0] - 1, offset[1] - 1
    out = []
    for c in contours:
        if cv2.contourArea(c) < 2.0:
            continue
        approx = cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        d = smooth_path(approx + np.array([ox, oy]))
        if d:
            out.append(d)
    return "".join(out)


def build_svgs(
    labels: np.ndarray,
    lines: np.ndarray,
    regions: list[dict[str, Any]],
    *,
    difficulty: str = "",
    title: str = "Coloriage",
) -> tuple[str, str, int]:
    """(coloring_svg, model_svg, zone_path_count)."""
    h, w = labels.shape
    slices = ndimage.find_objects(labels)
    zone_parts: list[tuple[dict[str, Any], str]] = []
    ink_parts: list[str] = []
    for region in regions:
        rid = region["id"]
        if rid - 1 >= len(slices) or slices[rid - 1] is None:
            continue
        sy, sx = slices[rid - 1]
        d = mask_to_path(labels[sy, sx] == rid, (sx.start, sy.start))
        if not d:
            continue
        if region["black_fill"]:
            ink_parts.append(d)
        else:
            zone_parts.append((region, d))
    line_d = mask_to_path(lines, (0, 0), epsilon=0.7)

    def doc(colored: bool, doc_title: str) -> str:
        head = (
            f'<svg xmlns="{SVG_NS}" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'data-generator="OpenDrawer" data-difficulty={quoteattr(difficulty)}>\n'
            f"<title>{doc_title}</title>\n"
        )
        zones = ['<g id="zones" stroke="none">']
        for region, d in zone_parts:
            fill = region["color_hex"] if colored else "#FFFFFF"
            zones.append(
                f'<path id="zone-{region["id"]}" class="zone" fill-rule="evenodd" fill="{fill}" '
                f'data-color="{region["color_hex"]}" data-color-name={quoteattr(region.get("color_name", ""))} '
                f'data-area="{region["area_px"]}" d="{d}"/>'
            )
        zones.append("</g>")
        ink = f'<g id="ink-fills" fill="{BLACK_HEX}" stroke="none">' + "".join(
            f'<path fill-rule="evenodd" d="{d}"/>' for d in ink_parts
        ) + "</g>"
        line = f'<g id="lines" fill="{BLACK_HEX}" stroke="none"><path fill-rule="evenodd" d="{line_d}"/></g>'
        return head + "\n".join(zones) + "\n" + ink + "\n" + line + "\n</svg>\n"

    return doc(False, title), doc(True, f"{title} — modèle colorié"), len(zone_parts)


class VectorizeSvg(BaseTool):
    name = "vectorize_svg"
    version = "1.0.0"
    tier = ToolTier.LAYOUT
    capability = "vectorize"
    provider = "opencv"
    stability = ToolStability.BETA
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    dependencies = ["python:cv2", "python:scipy"]
    install_instructions = "pip install opencv-python-headless scipy"
    capabilities = ["svg_export", "fillable_zones", "expected_colors"]
    best_for = ["impression nette à toute taille", "application de coloriage numérique"]

    input_schema = {
        "type": "object",
        "required": ["region_map", "output_dir"],
        "properties": {
            "region_map": {"type": "object"},
            "output_dir": {"type": "string"},
            "title": {"type": "string", "default": "Coloriage"},
        },
    }
    resource_profile = ResourceProfile(cpu_cores=1, ram_mb=1500)

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        started = time.monotonic()
        try:
            rm = inputs["region_map"]
            labels = decode_labels(load_rgb(rm["labels_path"]))
            lines = ~load_mask(rm["lines_path"])
            coloring, model, zone_count = build_svgs(
                labels, lines, rm["regions"], difficulty=rm.get("difficulty", ""), title=inputs.get("title", "Coloriage")
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"vectorize_svg failed: {exc}")
        out = Path(inputs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        coloring_path = out / "coloriage.svg"
        model_path = out / "modele.svg"
        coloring_path.write_text(coloring, encoding="utf-8")
        model_path.write_text(model, encoding="utf-8")
        return ToolResult(
            success=True,
            data={
                "coloring_svg": str(coloring_path),
                "model_svg": str(model_path),
                "zone_paths": zone_count,
                "size_bytes": coloring_path.stat().st_size,
            },
            artifacts=[str(coloring_path), str(model_path)],
            duration_seconds=time.monotonic() - started,
        )
