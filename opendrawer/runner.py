"""Deterministic end-to-end runner used by the CLI.

The agent normally drives the pipeline stage by stage (reading the director
skills). This runner executes the *same* tools in the *same* order and writes
the *same* checkpoints, so a project started by the CLI can be resumed or
reviewed by an agent and vice versa. Human gates are delegated to an
``Approver`` (interactive prompt, or auto-approve with ``--yes``).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from lib.checkpoint import (
    get_next_stage,
    init_project,
    read_checkpoint,
    write_checkpoint,
)
from lib.paths import PROJECTS_DIR
from tools.base_tool import ToolResult
from tools.tool_registry import registry

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
PAGE_PIPELINE = "coloring-page"
BOOK_PIPELINE = "coloring-book"


class StageError(RuntimeError):
    """A tool failed; the stage checkpoint was written as failed."""


class Approver(Protocol):
    def choose_cartoon(self, project_id: str, cartoon_set: dict[str, Any]) -> Optional[str]:
        """Return the approved candidate id, or None to stop and wait."""

    def approve(self, project_id: str, stage: str, summary: dict[str, Any]) -> bool:
        """Approve a gated stage (True) or stop and wait (False)."""


class AutoApprover:
    """``--yes``: accept the recommended candidate and every gate."""

    def choose_cartoon(self, project_id: str, cartoon_set: dict[str, Any]) -> Optional[str]:
        return cartoon_set.get("selected_id") or cartoon_set["candidates"][0]["id"]

    def approve(self, project_id: str, stage: str, summary: dict[str, Any]) -> bool:
        return True


@dataclass
class RunSettings:
    difficulty: str = "medium"
    style: str = "livre-jeunesse"
    backend: str = "auto"
    family: str = "auto"
    identity: str = "auto"
    candidates: int = 3
    seed: Optional[int] = None
    steps: Optional[int] = None  # override the style's diffusion steps
    color_source: str = "photo"
    commercial_safe: bool = False
    paper: str = "A4"
    dpi: int = 300
    margin_mm: float = 10.0
    thumbnail_ratio: float = 0.26
    neural_lineart: str = "auto"
    descriptions: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_marker(cls, marker: dict[str, Any]) -> "RunSettings":
        raw = marker.get("settings") or {}
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in known})


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text or "projet"


def unique_project_id(base: str, projects_dir: Path) -> str:
    candidate, i = base, 2
    while (projects_dir / candidate).exists():
        candidate = f"{base}-{i}"
        i += 1
    return candidate


def _save_artifact(project_dir: Path, name: str, data: dict[str, Any]) -> Path:
    path = project_dir / "artifacts" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _run_tool(name: str, inputs: dict[str, Any]) -> ToolResult:
    registry.ensure_discovered()
    tool = registry.get(name)
    if tool is None:
        raise StageError(f"Tool {name!r} is not registered")
    result = tool.execute(inputs)
    if not result.success:
        raise StageError(f"{name}: {result.error}")
    return result


Log = Callable[[str], None]


class PageRun:
    """One photo → one coloring page (pipeline ``coloring-page``)."""

    def __init__(
        self,
        project_id: str,
        approver: Approver,
        *,
        projects_dir: Path = PROJECTS_DIR,
        log: Log = print,
    ) -> None:
        self.project_id = project_id
        self.projects_dir = projects_dir
        self.project_dir = projects_dir / project_id
        self.approver = approver
        self.log = log
        marker = json.loads((self.project_dir / "project.json").read_text(encoding="utf-8"))
        self.settings = RunSettings.from_marker(marker)
        self.photo = marker["source_photo"]

    # ------------------------------------------------------------ creation

    @classmethod
    def create(
        cls,
        photo: str | Path,
        settings: RunSettings,
        approver: Approver,
        *,
        project_id: Optional[str] = None,
        title: Optional[str] = None,
        projects_dir: Path = PROJECTS_DIR,
        extra: Optional[dict[str, Any]] = None,
        log: Log = print,
    ) -> "PageRun":
        photo = Path(photo)
        if not photo.is_file():
            raise FileNotFoundError(f"Photo not found: {photo}")
        if photo.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image type: {photo.suffix}")
        pid = project_id or unique_project_id(slugify(photo.stem), projects_dir)
        project_dir = init_project(
            pid,
            title=title or photo.stem,
            pipeline_type=PAGE_PIPELINE,
            pipeline_dir=projects_dir,
            style=settings.style,
            extra={"settings": asdict(settings), **(extra or {})},
        )
        dest = project_dir / "input" / f"photo{photo.suffix.lower()}"
        shutil.copyfile(photo, dest)
        marker_path = project_dir / "project.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["source_photo"] = str(dest)
        marker_path.write_text(json.dumps(marker, indent=2, ensure_ascii=False), encoding="utf-8")
        return cls(pid, approver, projects_dir=projects_dir, log=log)

    # ------------------------------------------------------------ helpers

    def _checkpoint(self, stage: str, status: str, artifacts: dict[str, Any], **kwargs: Any) -> None:
        write_checkpoint(self.projects_dir, self.project_id, stage, status, artifacts, pipeline_type=PAGE_PIPELINE, **kwargs)

    def _artifact(self, stage: str, name: str) -> dict[str, Any]:
        cp = read_checkpoint(self.projects_dir, self.project_id, stage)
        if not cp or name not in cp["artifacts"]:
            raise StageError(f"Missing artifact {name!r} from stage {stage!r}")
        return cp["artifacts"][name]

    def _env(self) -> None:
        if self.settings.commercial_safe:
            os.environ["OPENDRAWER_COMMERCIAL_SAFE"] = "1"

    # ------------------------------------------------------------ stages

    def stage_intake(self) -> dict[str, Any]:
        d = self.project_dir / "assets" / "intake"
        det = _run_tool("subject_detect", {"image_path": self.photo})
        subjects = det.data["subjects"]
        seg = _run_tool(
            "bg_remove",
            {
                "input_path": self.photo,
                "mask_path": str(d / "foreground_mask.png"),
                "cutout_path": str(d / "cutout.png"),
                "subject_boxes": [s["bbox"] for s in subjects],
                "subject_kinds": [s["kind"] for s in subjects],
            },
        )
        det2 = _run_tool(
            "subject_detect",
            {
                "image_path": self.photo,
                "foreground_mask_path": seg.data["mask_path"],
                "mask_dir": str(d),
            },
        )
        pal = _run_tool("palette_extract", {"image_path": seg.data["cutout_path"], "mask_path": seg.data["mask_path"]})
        return {
            "photo_analysis": {
                "version": "1.0",
                "source_photo": self.photo,
                "image_size": det2.data["image_size"],
                "subjects": det2.data["subjects"],
                "cutout_path": seg.data["cutout_path"],
                "foreground_mask_path": seg.data["mask_path"],
                "palette": pal.data["palette"],
                "warnings": det2.data["warnings"] + seg.data["warnings"],
                "tools_used": det2.data["tools_used"] + [f"bg_remove:{seg.data['model']}", "palette_extract"],
            }
        }

    def stage_cartoon(self) -> dict[str, Any]:
        s = self.settings
        res = _run_tool(
            "cartoonize",
            {
                "photo_analysis": self._artifact("intake", "photo_analysis"),
                "output_dir": str(self.project_dir / "assets" / "cartoon"),
                "backend": s.backend,
                "style": s.style,
                "candidates": s.candidates,
                "seed": s.seed,
                "steps": s.steps,
                "family": s.family,
                "identity": s.identity,
                "commercial_safe": s.commercial_safe,
                "descriptions": s.descriptions,
            },
        )
        for warning in res.data.get("warnings", []):
            self.log(f"  ! {warning}")
        return {"cartoon_set": res.data["cartoon_set"]}

    def stage_line_art(self) -> dict[str, Any]:
        cartoon_set = self._artifact("cartoon", "cartoon_set")
        chosen = next(c for c in cartoon_set["candidates"] if c["id"] == cartoon_set["selected_id"])
        res = _run_tool(
            "lineart_extract",
            {
                "cartoon_path": chosen["path"],
                "foreground_mask_path": self._artifact("intake", "photo_analysis")["foreground_mask_path"],
                "output_dir": str(self.project_dir / "assets" / "line_art"),
                "difficulty": self.settings.difficulty,
                "neural": self.settings.neural_lineart,
                "commercial_safe": self.settings.commercial_safe,
            },
        )
        return {"line_art": res.data["line_art"]}

    def stage_regions(self) -> dict[str, Any]:
        res = _run_tool(
            "region_segment",
            {
                "line_art": self._artifact("line_art", "line_art"),
                "palette": self._artifact("intake", "photo_analysis")["palette"],
                "color_source": self.settings.color_source,
                "output_dir": str(self.project_dir / "assets" / "regions"),
            },
        )
        return {"region_map": res.data["region_map"]}

    def stage_page(self) -> dict[str, Any]:
        rm = self._artifact("regions", "region_map")
        out = self.project_dir / "output"
        vec = _run_tool("vectorize_svg", {"region_map": rm, "output_dir": str(out)})
        s = self.settings
        lay = _run_tool(
            "page_layout",
            {
                "region_map": rm,
                "coloring_svg": vec.data["coloring_svg"],
                "model_svg": vec.data["model_svg"],
                "output_dir": str(out),
                "paper": s.paper,
                "dpi": s.dpi,
                "margin_mm": s.margin_mm,
                "thumbnail_ratio": s.thumbnail_ratio,
            },
        )
        cp = lay.data["coloring_page"]
        qa = _run_tool(
            "coloring_qa",
            {"region_map": rm, "coloring_page": cp, "output_path": str(self.project_dir / "artifacts" / "qa_report.json")},
        )
        report = qa.data["qa_report"]
        cp["qa_status"] = report["status"]
        return {"coloring_page": cp, "qa_report": report}

    STAGES: dict[str, str] = {
        "intake": "stage_intake",
        "cartoon": "stage_cartoon",
        "line_art": "stage_line_art",
        "regions": "stage_regions",
        "page": "stage_page",
    }
    GATED = {"cartoon", "page"}

    # ------------------------------------------------------------ driver

    def _resolve_gate(self, stage: str, artifacts: dict[str, Any]) -> bool:
        """Ask the approver; on approval rewrite the stage as completed."""
        if stage == "cartoon":
            chosen = self.approver.choose_cartoon(self.project_id, artifacts["cartoon_set"])
            if not chosen:
                return False
            cs = dict(artifacts["cartoon_set"])
            if chosen != cs.get("selected_id"):
                cs["selection_reason"] = "chosen by the user"
            else:
                cs["selection_reason"] = "recommended candidate approved by the user"
            cs["selected_id"] = chosen
            artifacts = {**artifacts, "cartoon_set": cs}
        else:
            if not self.approver.approve(self.project_id, stage, artifacts):
                return False
        for name, data in artifacts.items():
            _save_artifact(self.project_dir, name, data)
        self._checkpoint(stage, "completed", artifacts, human_approval_required=True, human_approved=True)
        return True

    def run(self) -> str:
        """Run until done or a gate is left pending. Returns the final status."""
        self._env()
        while True:
            stage = get_next_stage(self.projects_dir, self.project_id, PAGE_PIPELINE)
            if stage is None:
                return "completed"
            existing = read_checkpoint(self.projects_dir, self.project_id, stage)
            if existing and existing["status"] == "awaiting_human":
                self.log(f"[{self.project_id}] {stage}: en attente de validation")
                if not self._resolve_gate(stage, existing["artifacts"]):
                    return "awaiting_human"
                continue

            self.log(f"[{self.project_id}] {stage}…")
            self._checkpoint(stage, "in_progress", {})
            try:
                artifacts = getattr(self, self.STAGES[stage])()
            except Exception as exc:
                self._checkpoint(stage, "failed", {}, error=str(exc)[:2000])
                raise StageError(f"[{self.project_id}] stage {stage} failed: {exc}") from exc

            for name, data in artifacts.items():
                _save_artifact(self.project_dir, name, data)
            if stage in self.GATED:
                self._checkpoint(stage, "awaiting_human", artifacts, human_approval_required=True)
                if not self._resolve_gate(stage, artifacts):
                    return "awaiting_human"
            else:
                self._checkpoint(stage, "completed", artifacts)


class BookRun:
    """A folder of photos → a multi-page book (pipeline ``coloring-book``)."""

    def __init__(self, book_id: str, approver: Approver, *, projects_dir: Path = PROJECTS_DIR, log: Log = print) -> None:
        self.book_id = book_id
        self.projects_dir = projects_dir
        self.project_dir = projects_dir / book_id
        self.approver = approver
        self.log = log
        marker = json.loads((self.project_dir / "project.json").read_text(encoding="utf-8"))
        self.settings = RunSettings.from_marker(marker)
        self.source_dir = marker["source_dir"]

    @classmethod
    def create(
        cls,
        source_dir: str | Path,
        settings: RunSettings,
        approver: Approver,
        *,
        book_id: Optional[str] = None,
        title: Optional[str] = None,
        projects_dir: Path = PROJECTS_DIR,
        log: Log = print,
    ) -> "BookRun":
        source_dir = Path(source_dir)
        if not source_dir.is_dir():
            raise NotADirectoryError(f"Not a folder: {source_dir}")
        bid = book_id or unique_project_id(f"livre-{slugify(source_dir.name)}", projects_dir)
        init_project(
            bid,
            title=title or f"Livre {source_dir.name}",
            pipeline_type=BOOK_PIPELINE,
            pipeline_dir=projects_dir,
            style=settings.style,
            extra={"settings": asdict(settings), "source_dir": str(source_dir.resolve())},
        )
        return cls(bid, approver, projects_dir=projects_dir, log=log)

    def _checkpoint(self, stage: str, status: str, artifacts: dict[str, Any], **kwargs: Any) -> None:
        write_checkpoint(self.projects_dir, self.book_id, stage, status, artifacts, pipeline_type=BOOK_PIPELINE, **kwargs)

    def _manifest(self) -> dict[str, Any]:
        for stage in ("assemble", "pages", "collect"):
            cp = read_checkpoint(self.projects_dir, self.book_id, stage)
            if cp and "book_manifest" in cp["artifacts"]:
                return cp["artifacts"]["book_manifest"]
        raise StageError("book_manifest not found")

    def stage_collect(self) -> dict[str, Any]:
        photos = sorted(
            p for p in Path(self.source_dir).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not photos:
            raise StageError(f"No photo found in {self.source_dir}")
        pages = [
            {
                "index": i,
                "project_id": f"{self.book_id}-p{i:02d}",
                "source_photo": str(p),
                "status": "pending",
            }
            for i, p in enumerate(photos, start=1)
        ]
        return {
            "book_manifest": {
                "version": "1.0",
                "book_id": self.book_id,
                "title": self.book_id,
                "difficulty": self.settings.difficulty,
                "source_dir": self.source_dir,
                "pages": pages,
            }
        }

    def stage_pages(self) -> dict[str, Any]:
        manifest = json.loads(json.dumps(self._manifest()))
        for page in manifest["pages"]:
            if page["status"] == "completed":
                continue
            pid = page["project_id"]
            try:
                if (self.projects_dir / pid / "project.json").exists():
                    run = PageRun(pid, self.approver, projects_dir=self.projects_dir, log=self.log)
                else:
                    run = PageRun.create(
                        page["source_photo"],
                        self.settings,
                        self.approver,
                        project_id=pid,
                        title=f"{self.book_id} — page {page['index']}",
                        projects_dir=self.projects_dir,
                        extra={"book_id": self.book_id, "page_index": page["index"]},
                        log=self.log,
                    )
                status = run.run()
            except StageError as exc:
                self.log(f"  ! {exc}")
                page["status"] = "failed"
                page["note"] = str(exc)[:500]
                continue
            page["status"] = status
            if status == "completed":
                cp = read_checkpoint(self.projects_dir, pid, "page")
                page["page_png"] = cp["artifacts"]["coloring_page"]["page_png"]
                page["page_svg"] = cp["artifacts"]["coloring_page"]["page_svg"]
        return {"book_manifest": manifest}

    def stage_assemble(self) -> dict[str, Any]:
        res = _run_tool(
            "book_assemble",
            {"book_manifest": self._manifest(), "output_dir": str(self.project_dir / "output" / "livre")},
        )
        return {"book_manifest": res.data["book_manifest"]}

    STAGES = {"collect": "stage_collect", "pages": "stage_pages", "assemble": "stage_assemble"}
    GATED = {"pages", "assemble"}

    def run(self) -> str:
        if self.settings.commercial_safe:
            os.environ["OPENDRAWER_COMMERCIAL_SAFE"] = "1"
        while True:
            stage = get_next_stage(self.projects_dir, self.book_id, BOOK_PIPELINE)
            if stage is None:
                return "completed"
            existing = read_checkpoint(self.projects_dir, self.book_id, stage)
            if existing and existing["status"] == "awaiting_human":
                artifacts = existing["artifacts"]
            else:
                self.log(f"[{self.book_id}] {stage}…")
                self._checkpoint(stage, "in_progress", {})
                try:
                    artifacts = getattr(self, self.STAGES[stage])()
                except Exception as exc:
                    self._checkpoint(stage, "failed", {}, error=str(exc)[:2000])
                    raise StageError(f"[{self.book_id}] stage {stage} failed: {exc}") from exc
                _save_artifact(self.project_dir, "book_manifest", artifacts["book_manifest"])

            if stage == "pages":
                pending = [p for p in artifacts["book_manifest"]["pages"] if p["status"] not in ("completed", "failed", "skipped")]
                if pending:
                    # Some pages wait for their own gate: keep the book stage open.
                    self._checkpoint(stage, "in_progress", {})
                    return "awaiting_human"
            if stage in self.GATED:
                self._checkpoint(stage, "awaiting_human", artifacts, human_approval_required=True)
                if not self.approver.approve(self.book_id, stage, artifacts):
                    return "awaiting_human"
                self._checkpoint(stage, "completed", artifacts, human_approval_required=True, human_approved=True)
            else:
                self._checkpoint(stage, "completed", artifacts)
