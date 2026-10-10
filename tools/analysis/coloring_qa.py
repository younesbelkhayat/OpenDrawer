"""Deterministic quality checks for a coloring page.

Runs before the agent's own visual review and before the human gate. Each
check returns pass/warn/fail with the measured value and its threshold, so
the agent can decide what to re-run (e.g. a leak → stronger gap closing,
too many zones → lower max_regions).
"""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy import ndimage

from lib.difficulty import get_preset
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
from tools.coloring.region_segment import decode_labels


def _check(cid: str, status: str, message: str, value: Any = None, threshold: Any = None) -> dict[str, Any]:
    out = {"id": cid, "status": status, "message": message}
    if value is not None:
        out["value"] = value
    if threshold is not None:
        out["threshold"] = threshold
    return out


def check_region_map(rm: dict[str, Any], preset) -> list[dict[str, Any]]:
    checks = []
    labels = decode_labels(load_rgb(rm["labels_path"]))
    coloring = load_rgb(rm["coloring_png"])
    regions = rm["regions"]
    zones = [r for r in regions if not r["black_fill"]]
    subject_area = float((labels > 0).sum()) or 1.0

    count = len(zones)
    if count == 0:
        checks.append(_check("zone_count", "fail", "Aucune zone à colorier", count))
    elif preset.min_regions <= count <= preset.max_regions:
        checks.append(_check("zone_count", "pass", f"{count} zones à colorier", count, [preset.min_regions, preset.max_regions]))
    else:
        checks.append(
            _check(
                "zone_count",
                "warn",
                f"{count} zones, hors de la plage {preset.min_regions}-{preset.max_regions} prévue pour « {preset.label_fr} »",
                count,
                [preset.min_regions, preset.max_regions],
            )
        )

    min_area = preset.min_region_frac * subject_area
    small = [r["id"] for r in zones if r["area_px"] < 0.5 * min_area]
    checks.append(
        _check(
            "min_zone_area",
            "warn" if small else "pass",
            f"{len(small)} zone(s) trop petite(s) pour être coloriée(s)" if small else "Toutes les zones sont assez grandes",
            len(small),
            round(0.5 * min_area),
        )
    )

    # Leaks: a zone's white interior must never connect to the page background.
    white = np.all(coloring > 200, axis=2)
    comps, _ = ndimage.label(white)
    border_comps = set(np.unique(np.concatenate([comps[0], comps[-1], comps[:, 0], comps[:, -1]]))) - {0}
    leaking = []
    for r in zones:
        mask = (labels == r["id"]) & white
        if not mask.any():
            continue
        if set(np.unique(comps[mask])) & border_comps:
            leaking.append(r["id"])
    checks.append(
        _check(
            "closed_zones",
            "fail" if leaking else "pass",
            f"Zones ouvertes vers le fond : {leaking}" if leaking else "Toutes les zones sont fermées",
            len(leaking),
            0,
        )
    )

    # Stroke width: twice the median distance-to-background on the stroke skeleton.
    lines = ~load_mask(rm["lines_path"])
    if lines.any():
        from skimage.morphology import skeletonize

        dt = ndimage.distance_transform_edt(lines)
        skel = skeletonize(lines)
        width = float(2 * np.median(dt[skel])) if skel.any() else 0.0
        ok = width >= 0.75 * preset.line_px
        checks.append(
            _check(
                "line_width",
                "pass" if ok else "warn",
                f"Épaisseur médiane des traits : {width:.1f}px",
                round(width, 1),
                round(0.75 * preset.line_px, 1),
            )
        )

    # White background outside the subject.
    outside = labels == 0
    outside = cv2.erode(outside.astype(np.uint8), np.ones((2 * preset.outline_px + 1,) * 2, np.uint8)) > 0
    dirty = float(np.mean(np.any(coloring[outside] < 230, axis=1))) if outside.any() else 0.0
    checks.append(
        _check(
            "white_background",
            "pass" if dirty < 0.001 else "warn",
            "Fond blanc" if dirty < 0.001 else f"{dirty:.2%} du fond n'est pas blanc",
            round(dirty, 5),
            0.001,
        )
    )

    colors = {r["color_hex"] for r in zones}
    checks.append(
        _check(
            "model_colors",
            "pass" if len(colors) >= 2 else "warn",
            f"{len(colors)} couleur(s) dans le modèle",
            len(colors),
            2,
        )
    )
    return checks


def check_page(cp: dict[str, Any], rm: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    for key in ("page_png", "page_svg", "coloring_png", "coloring_svg", "model_png", "model_svg"):
        if not Path(cp[key]).is_file():
            checks.append(_check(f"file_{key}", "fail", f"Fichier manquant : {cp[key]}"))

    for key in ("coloring_svg", "model_svg", "page_svg"):
        try:
            root = ET.parse(cp[key]).getroot()
            if key == "coloring_svg":
                n = sum(1 for el in root.iter() if el.attrib.get("class") == "zone")
                expected = sum(1 for r in rm["regions"] if not r["black_fill"])
                checks.append(
                    _check(
                        "svg_zones",
                        "pass" if n == expected else "warn",
                        f"{n} zones remplissables dans le SVG (attendu {expected})",
                        n,
                        expected,
                    )
                )
        except (ET.ParseError, OSError) as exc:
            checks.append(_check(f"svg_valid_{key}", "fail", f"SVG invalide ({key}) : {exc}"))

    # Thumbnail must not cover the drawing's ink.
    drawing = load_rgb(cp["coloring_png"])
    dx, dy, dw, dh = cp["drawing_box"]
    tx, ty, tw, th = cp["thumbnail_box"]
    ix0, iy0 = max(dx, tx), max(dy, ty)
    ix1, iy1 = min(dx + dw, tx + tw), min(dy + dh, ty + th)
    overlap = 0
    if ix1 > ix0 and iy1 > iy0:
        sx, sy = drawing.shape[1] / dw, drawing.shape[0] / dh
        sub = drawing[int((iy0 - dy) * sy):int((iy1 - dy) * sy), int((ix0 - dx) * sx):int((ix1 - dx) * sx)]
        overlap = int(np.sum(np.any(sub < 128, axis=2))) if sub.size else 0
    checks.append(
        _check(
            "thumbnail_overlap",
            "fail" if overlap else "pass",
            "La vignette recouvre le dessin" if overlap else "Vignette séparée du dessin",
            overlap,
            0,
        )
    )

    pw, ph = cp["page_size"]["width"], cp["page_size"]["height"]
    coverage = (dw * dh) / float(pw * ph)
    checks.append(
        _check(
            "drawing_size",
            "pass" if coverage >= 0.3 else "warn",
            f"Le dessin occupe {coverage:.0%} de la page",
            round(coverage, 3),
            0.3,
        )
    )
    return checks


class ColoringQA(BaseTool):
    name = "coloring_qa"
    version = "1.0.0"
    tier = ToolTier.ANALYZE
    capability = "quality_check"
    provider = "opendrawer"
    stability = ToolStability.BETA
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    dependencies = ["python:cv2", "python:scipy", "python:skimage"]
    install_instructions = "pip install opencv-python-headless scipy scikit-image"
    capabilities = ["closed_zone_check", "zone_count_check", "line_width_check", "layout_check"]

    input_schema = {
        "type": "object",
        "required": ["region_map"],
        "properties": {
            "region_map": {"type": "object"},
            "coloring_page": {"type": "object", "description": "Adds file/SVG/layout checks when given"},
            "difficulty": {"type": "string"},
            "output_path": {"type": "string", "description": "Optional JSON report path"},
        },
    }
    resource_profile = ResourceProfile(cpu_cores=1, ram_mb=1500)

    def get_status(self) -> ToolStatus:
        try:
            import scipy  # noqa: F401
            import skimage  # noqa: F401

            return ToolStatus.AVAILABLE
        except ImportError:
            return ToolStatus.UNAVAILABLE

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        started = time.monotonic()
        try:
            rm = inputs["region_map"]
            preset = get_preset(inputs.get("difficulty") or rm.get("difficulty"))
            checks = check_region_map(rm, preset)
            if inputs.get("coloring_page"):
                checks += check_page(inputs["coloring_page"], rm)
        except Exception as exc:
            return ToolResult(success=False, error=f"coloring_qa failed: {exc}")

        statuses = {c["status"] for c in checks}
        status = "fail" if "fail" in statuses else "warn" if "warn" in statuses else "pass"
        problems = [c["message"] for c in checks if c["status"] != "pass"]
        report = {
            "version": "1.0",
            "status": status,
            "difficulty": preset.name,
            "checks": checks,
            "summary": "Tout est conforme" if not problems else "; ".join(problems),
        }
        if inputs.get("output_path"):
            path = Path(inputs["output_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return ToolResult(success=True, data={"qa_report": report}, duration_seconds=time.monotonic() - started)
