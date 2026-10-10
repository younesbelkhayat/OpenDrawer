"""Assemble validated coloring pages into a numbered book folder.

Copies each completed page (PNG + SVG) as ``page_01.png`` / ``page_01.svg``…
into the book's output folder and draws a contact sheet (``sommaire.png``)
so the whole book can be checked at a glance.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lib.imaging import load_rgb, save_rgb
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


def contact_sheet(pages: list[Path], thumb_w: int = 420, cols: int = 4, gap: int = 30) -> np.ndarray:
    thumbs = []
    for i, p in enumerate(pages, start=1):
        img = load_rgb(p)
        th = round(img.shape[0] * thumb_w / img.shape[1])
        t = cv2.resize(img, (thumb_w, th), interpolation=cv2.INTER_AREA)
        cv2.rectangle(t, (0, 0), (thumb_w - 1, th - 1), (180, 180, 180), 2)
        cv2.putText(t, str(i), (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (40, 40, 40), 3, cv2.LINE_AA)
        thumbs.append(t)
    if not thumbs:
        return np.full((200, 200, 3), 255, np.uint8)
    cols = min(cols, len(thumbs))
    rows = math.ceil(len(thumbs) / cols)
    th = max(t.shape[0] for t in thumbs)
    sheet = np.full((rows * (th + gap) + gap, cols * (thumb_w + gap) + gap, 3), 255, np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        y, x = gap + r * (th + gap), gap + c * (thumb_w + gap)
        sheet[y:y + t.shape[0], x:x + thumb_w] = t
    return sheet


class BookAssemble(BaseTool):
    name = "book_assemble"
    version = "1.0.0"
    tier = ToolTier.LAYOUT
    capability = "book"
    provider = "opendrawer"
    stability = ToolStability.BETA
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    dependencies = ["python:cv2"]
    install_instructions = "pip install opencv-python-headless"
    capabilities = ["multi_page_book", "contact_sheet"]

    input_schema = {
        "type": "object",
        "required": ["book_manifest", "output_dir"],
        "properties": {
            "book_manifest": {"type": "object"},
            "output_dir": {"type": "string"},
            "include_statuses": {
                "type": "array",
                "items": {"type": "string"},
                "default": ["completed"],
                "description": "Page statuses to include (default: approved pages only)",
            },
        },
    }
    resource_profile = ResourceProfile(cpu_cores=1, ram_mb=1000)

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        started = time.monotonic()
        manifest = json.loads(json.dumps(inputs["book_manifest"]))
        include = set(inputs.get("include_statuses") or ["completed"])
        out = Path(inputs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)

        copied: list[Path] = []
        skipped = []
        number = 0
        for page in sorted(manifest["pages"], key=lambda p: p["index"]):
            src_png, src_svg = page.get("page_png"), page.get("page_svg")
            if page["status"] not in include or not src_png or not Path(src_png).is_file():
                skipped.append(page["index"])
                continue
            number += 1
            dst_png = out / f"page_{number:02d}.png"
            shutil.copyfile(src_png, dst_png)
            copied.append(dst_png)
            page["book_page_png"] = str(dst_png)
            if src_svg and Path(src_svg).is_file():
                dst_svg = out / f"page_{number:02d}.svg"
                shutil.copyfile(src_svg, dst_svg)
                page["book_page_svg"] = str(dst_svg)

        if not copied:
            return ToolResult(success=False, error="book_assemble: no completed page to assemble")
        sheet_path = save_rgb(contact_sheet(copied), out / "sommaire.png")
        manifest["output_dir"] = str(out)
        manifest["contact_sheet"] = str(sheet_path)
        return ToolResult(
            success=True,
            data={"book_manifest": manifest, "page_count": len(copied), "skipped_pages": skipped},
            artifacts=[str(p) for p in copied] + [str(sheet_path)],
            duration_seconds=time.monotonic() - started,
        )
