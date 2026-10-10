"""BoardState derivation — turn a project directory into renderable state.

Everything here is read-only and defensive: a malformed JSON file, a missing
artifact, or a half-written checkpoint must degrade the board, never crash it
(design principle: "never block, never break").
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from lib.events import read_events
from lib.paths import PROJECTS_DIR, REPO_ROOT  # single source of truth (env-overridable)

MEDIA_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}

# Directories inside a project we never scan for media (build noise).
SCAN_EXCLUDE = {"node_modules", ".git", "__pycache__", "history", ".cache"}

# Fallback rail when the manifest is unknown (coloring-page order).
FALLBACK_STAGES = ["intake", "cartoon", "line_art", "regions", "page"]

# How long (seconds) without filesystem activity before a board reads "idle".
LIVE_WINDOW_SECONDS = 5 * 60

# An in_progress stage with no filesystem activity for this long is flagged
# as possibly stalled (F-05: a wedged agent must be visible, not silent —
# heartbeat checkpoints and tool events both reset the clock).
STALL_WINDOW_SECONDS = 10 * 60


def _read_json(path: Path) -> Optional[dict]:
    """Read a JSON file, returning None on any failure."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None


def _rel(project_dir: Path, path: Path) -> str:
    """Project-relative POSIX path for media URLs."""
    try:
        return path.resolve().relative_to(Path(project_dir).resolve()).as_posix()
    except (ValueError, OSError):
        return path.name


# ---------------------------------------------------------------------------
# Pipeline / stages
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _load_pipeline_meta(pipeline_type: Optional[str]) -> dict[str, Any]:
    """Stage order + gate flags from the manifest; graceful fallback."""
    if pipeline_type and pipeline_type != "unknown":
        try:
            from lib.pipeline_loader import load_pipeline
            manifest = load_pipeline(pipeline_type)
            stages = [
                {
                    "name": s["name"],
                    "gated": bool(s.get("human_approval_default", False)),
                    "produces": [
                        str(name) for name in (s.get("produces") or [])
                        if isinstance(name, str) and name
                    ],
                }
                for s in manifest.get("stages", [])
                if isinstance(s, dict) and s.get("name")
            ]
            if stages:
                return {
                    "pipeline_type": pipeline_type,
                    "stages": stages,
                    "known": True,
                }
        except Exception:
            pass
    return {
        "pipeline_type": pipeline_type or "unknown",
        "stages": [{"name": s, "gated": False, "produces": []} for s in FALLBACK_STAGES],
        "known": False,
    }


def _resolve_artifact(project_dir: Path, value: Any) -> Optional[dict]:
    """Checkpoint artifacts may be inline dicts or path strings — resolve both.

    Path references are only followed INSIDE the project directory: a
    checkpoint must not be able to pull arbitrary JSON from elsewhere on
    disk onto the board (F-04).
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        p = Path(value)
        if not p.is_absolute():
            p = project_dir / value
        try:
            p.resolve().relative_to(Path(project_dir).resolve())
        except (ValueError, OSError):
            return None
        return _read_json(p)
    return None


def _collect_checkpoints(project_dir: Path) -> dict[str, dict]:
    """Current checkpoint per stage (raw dicts, unvalidated by design)."""
    out: dict[str, dict] = {}
    for path in sorted(project_dir.glob("checkpoint_*.json")):
        stage = path.stem[len("checkpoint_"):]
        data = _read_json(path)
        if data is not None:
            data["_mtime"] = path.stat().st_mtime
            out[stage] = data
    return out


def _collect_history(project_dir: Path) -> dict[str, list[dict]]:
    """Archived checkpoint versions per stage (oldest first)."""
    history_dir = project_dir / "history"
    out: dict[str, list[dict]] = {}
    if not history_dir.is_dir():
        return out
    for path in sorted(history_dir.glob("checkpoint_*.json")):
        m = re.match(r"checkpoint_(.+?)_\d", path.stem)
        stage = m.group(1) if m else path.stem[len("checkpoint_"):]
        data = _read_json(path)
        if data is not None:
            out.setdefault(stage, []).append(data)
    return out


def _build_stage_rail(
    pipeline_meta: dict,
    checkpoints: dict[str, dict],
    history: dict[str, list[dict]],
) -> list[dict]:
    """One entry per manifest stage with derived status + gate audit."""
    rail = []
    manifest_stage_names = {s["name"] for s in pipeline_meta["stages"]}
    for stage_def in pipeline_meta["stages"]:
        name = stage_def["name"]
        cp = checkpoints.get(name)
        versions = history.get(name, [])
        status = cp.get("status") if cp else "pending"
        entry: dict[str, Any] = {
            "name": name,
            "gated": stage_def["gated"],
            "produces": list(stage_def.get("produces") or []),
            "status": status or "pending",
            "timestamp": cp.get("timestamp") if cp else None,
            "review": cp.get("review") if cp else None,
            "error": cp.get("error") if cp else None,
            "human_approved": cp.get("human_approved") if cp else None,
            "partial_progress": (cp.get("metadata") or {}).get("partial_progress") if cp else None,
            "versions": len(versions) + (1 if cp else 0),
            # Chronological status trail (history + current) — powers replay.
            "history_entries": (
                [{"status": v.get("status"), "timestamp": v.get("timestamp")} for v in versions]
                + ([{"status": cp.get("status"), "timestamp": cp.get("timestamp")}] if cp else [])
            ),
        }
        # Gate audit: a gated stage that completed without ever passing
        # through awaiting_human (current or archived) was gate-skipped.
        if (
            stage_def["gated"]
            and cp is not None
            and cp.get("status") == "completed"
        ):
            saw_wait = any(v.get("status") == "awaiting_human" for v in versions)
            approved = bool(cp.get("human_approved"))
            entry["gate_skipped"] = not (saw_wait or approved)
        rail.append(entry)

    # Checkpoints for stages the manifest doesn't declare (legacy runs,
    # pipeline mismatch) still deserve a slot — at their canonical position
    # in the pipeline, not dangling at the end.
    canon = {name: i for i, name in enumerate(FALLBACK_STAGES)}
    for name, cp in checkpoints.items():
        if name in manifest_stage_names:
            continue
        entry = {
            "name": name,
            "gated": False,
            "produces": [
                str(artifact_name)
                for artifact_name in (cp.get("artifacts") or {})
                if isinstance(artifact_name, str) and artifact_name
            ],
            "status": cp.get("status") or "unknown",
            "timestamp": cp.get("timestamp"),
            "review": cp.get("review"),
            "error": cp.get("error"),
            "human_approved": cp.get("human_approved"),
            "partial_progress": None,
            "versions": 1 + len(history.get(name, [])),
            "undeclared": True,
        }
        pos = canon.get(name)
        if pos is None:
            rail.append(entry)  # truly unknown name — end of rail
            continue
        insert_at = len(rail)
        for i, existing in enumerate(rail):
            existing_pos = canon.get(existing["name"])
            if existing_pos is not None and existing_pos > pos:
                insert_at = i
                break
        rail.insert(insert_at, entry)
    return rail


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

ARTIFACT_FILES = {
    "photo_analysis": "photo_analysis.json",
    "cartoon_set": "cartoon_set.json",
    "line_art": "line_art.json",
    "region_map": "region_map.json",
    "coloring_page": "coloring_page.json",
    "qa_report": "qa_report.json",
    "book_manifest": "book_manifest.json",
    "decision_log": "decision_log.json",
}


def _collect_artifacts(project_dir: Path, checkpoints: dict[str, dict]) -> dict[str, dict]:
    """Artifacts from artifacts/*.json, backfilled from checkpoint payloads."""
    artifacts: dict[str, dict] = {}
    art_dir = project_dir / "artifacts"
    for name, filename in ARTIFACT_FILES.items():
        data = _read_json(art_dir / filename)
        if data is not None:
            artifacts[name] = data
    # decision_log is merged at the project root by lib/checkpoint.py
    if "decision_log" not in artifacts:
        data = _read_json(project_dir / "decision_log.json")
        if data is not None:
            artifacts["decision_log"] = data
    # Backfill from checkpoint-embedded artifacts.
    for cp in checkpoints.values():
        for name, value in (cp.get("artifacts") or {}).items():
            if name not in artifacts:
                resolved = _resolve_artifact(project_dir, value)
                if resolved is not None:
                    artifacts[name] = resolved
    return artifacts


# ---------------------------------------------------------------------------
# Step gallery (photo → cartoon → lines → zones → page)
# ---------------------------------------------------------------------------

def _inside(project_dir: Path, raw: Any) -> Optional[str]:
    """Project-relative path of an existing file inside the project, else None."""
    if not isinstance(raw, str) or not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = project_dir / raw
    try:
        rel = p.resolve().relative_to(Path(project_dir).resolve())
    except (ValueError, OSError):
        return None
    return rel.as_posix() if p.is_file() else None


def _image(project_dir: Path, raw: Any, label: str, **extra: Any) -> Optional[dict]:
    rel = _inside(project_dir, raw)
    if rel is None:
        return None
    return {"path": rel, "label": label, **{k: v for k, v in extra.items() if v is not None}}


def _build_steps(project_dir: Path, artifacts: dict[str, dict], events: list[dict]) -> list[dict]:
    """One gallery per stage, from the artifacts each stage produced."""
    steps: list[dict] = []
    running = {ev.get("tool") for ev in events[-30:] if ev.get("event") == "start"}
    running -= {ev.get("tool") for ev in events[-30:] if ev.get("event") in ("finish", "error")}

    pa = artifacts.get("photo_analysis") or {}
    images = [
        _image(project_dir, pa.get("source_photo"), "photo"),
        _image(project_dir, pa.get("cutout_path"), "détourage"),
    ]
    if not pa:
        for candidate in sorted((project_dir / "input").glob("photo.*")) if (project_dir / "input").is_dir() else []:
            images.append(_image(project_dir, str(candidate), "photo"))
    facts = []
    if pa:
        subjects = pa.get("subjects") or []
        facts.append(f"{len(subjects)} sujet(s) : " + ", ".join(s.get("label", s.get("kind", "?")) for s in subjects))
        facts.extend(pa.get("warnings") or [])
    steps.append({
        "stage": "intake",
        "title": "Photo",
        "images": [i for i in images if i],
        "palette": [c.get("hex") for c in (pa.get("palette") or [])][:16],
        "facts": facts,
    })

    cs = artifacts.get("cartoon_set") or {}
    cands = []
    for c in cs.get("candidates") or []:
        img = _image(
            project_dir,
            c.get("path"),
            c.get("id", "?"),
            score=(c.get("score") or {}).get("total"),
            selected=c.get("id") == cs.get("selected_id"),
        )
        if img:
            cands.append(img)
    steps.append({
        "stage": "cartoon",
        "title": "Cartoon",
        "images": cands,
        "facts": [
            f for f in (
                f"backend {cs.get('backend')}" if cs.get("backend") else None,
                f"modèle {cs.get('model_family')}" if cs.get("model_family") else None,
                f"ressemblance {cs.get('identity_mode')}" if cs.get("identity_mode") else None,
                cs.get("selection_reason"),
            ) if f
        ],
        "generating": bool(running & {"cartoonize", "diffusers_cartoonize", "comfyui_cartoonize", "opencv_cartoonize"}),
    })

    la = artifacts.get("line_art") or {}
    steps.append({
        "stage": "line_art",
        "title": "Traits",
        "images": [i for i in (
            _image(project_dir, la.get("line_mask_path"), "traits bruts"),
            _image(project_dir, la.get("flat_path"), "aplats"),
        ) if i],
        "facts": [", ".join(la.get("methods") or [])] if la.get("methods") else [],
    })

    rm = artifacts.get("region_map") or {}
    steps.append({
        "stage": "regions",
        "title": "Zones",
        "images": [i for i in (
            _image(project_dir, rm.get("coloring_png"), "zones"),
            _image(project_dir, rm.get("preview_png"), "aperçu des zones"),
            _image(project_dir, rm.get("model_png"), "modèle"),
        ) if i],
        "facts": [
            f"{rm.get('region_count')} zones · {rm.get('black_fill_count')} détails noircis · difficulté {rm.get('difficulty')}"
        ] if rm else [],
        "palette": [c.get("hex") for c in (rm.get("palette_used") or [])][:24],
    })

    cp = artifacts.get("coloring_page") or {}
    files = []
    for key, label in (("page_svg", "page.svg"), ("coloring_svg", "coloriage.svg"), ("model_svg", "modele.svg"),
                       ("coloring_png", "coloriage.png"), ("model_png", "modele.png")):
        rel = _inside(project_dir, cp.get(key))
        if rel:
            files.append({"path": rel, "label": label})
    steps.append({
        "stage": "page",
        "title": "Page",
        "images": [i for i in (_image(project_dir, cp.get("page_png"), "page A4"),) if i],
        "files": files,
        "facts": [f"{cp.get('paper')} · {cp.get('dpi')} dpi · mise en page {cp.get('layout_mode')}"] if cp else [],
    })
    return steps


def _build_book(project_dir: Path, artifacts: dict[str, dict]) -> Optional[dict]:
    manifest = artifacts.get("book_manifest")
    if not manifest:
        return None
    pages = []
    for page in manifest.get("pages") or []:
        entry = {
            "index": page.get("index"),
            "project_id": page.get("project_id"),
            "status": page.get("status"),
            "note": page.get("note"),
            "poster_project": None,
            "poster": None,
        }
        own = _inside(project_dir, page.get("book_page_png"))
        if own:
            entry["poster_project"], entry["poster"] = project_dir.name, own
        elif page.get("project_id") and page.get("page_png"):
            sub_dir = project_dir.parent / page["project_id"]
            rel = _inside(sub_dir, page["page_png"])
            if rel:
                entry["poster_project"], entry["poster"] = page["project_id"], rel
        pages.append(entry)
    return {
        "pages": pages,
        "contact_sheet": _inside(project_dir, manifest.get("contact_sheet")),
        "output_dir": manifest.get("output_dir"),
    }


def _scan_images(project_dir: Path) -> list[dict]:
    """Loose images (degraded view when a project has no artifacts yet)."""
    found = []
    for rel_dir in ("output", "assets/regions", "assets/cartoon", "input"):
        d = project_dir / rel_dir
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in MEDIA_IMAGE_EXT - {".svg"}:
                found.append({"path": _rel(project_dir, f)})
    return found[:24]


def _find_poster(project_dir: Path, state: dict) -> Optional[str]:
    """Most representative image for the library card."""
    by_stage = {s["stage"]: s for s in state.get("steps") or []}
    for stage in ("page", "regions", "cartoon", "intake"):
        images = (by_stage.get(stage) or {}).get("images") or []
        selected = [i for i in images if i.get("selected")]
        if selected:
            return selected[0]["path"]
        if images:
            return images[0]["path"]
    book = state.get("book") or {}
    if book.get("contact_sheet"):
        return book["contact_sheet"]
    images = state.get("images") or []
    return images[0]["path"] if images else None


def _last_activity(project_dir: Path) -> float:
    """Most recent mtime among state-bearing files (bounded scan)."""
    latest = 0.0
    try:
        candidates = list(project_dir.glob("checkpoint_*.json"))
        candidates.append(project_dir / "events.jsonl")
        art = project_dir / "artifacts"
        if art.is_dir():
            candidates.extend(art.glob("*.json"))
        for p in candidates:
            try:
                latest = max(latest, p.stat().st_mtime)
            except OSError:
                continue
    except OSError:
        pass
    return latest


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_board_state(project_dir: Path) -> dict[str, Any]:
    """Full BoardState for one project. Never raises."""
    project_dir = Path(project_dir)
    project_id = project_dir.name

    marker = _read_json(project_dir / "project.json") or {}
    meta_json = _read_json(project_dir / "meta.json") or {}

    checkpoints = _collect_checkpoints(project_dir)
    history = _collect_history(project_dir)

    pipeline_type = marker.get("pipeline_type")
    if not pipeline_type:
        for cp in checkpoints.values():
            pt = cp.get("pipeline_type")
            if pt and pt != "unknown":
                pipeline_type = pt
                break
    pipeline_meta = _load_pipeline_meta(pipeline_type)

    artifacts = _collect_artifacts(project_dir, checkpoints)
    events = read_events(project_dir, limit=250)
    stages = _build_stage_rail(pipeline_meta, checkpoints, history)
    is_book = pipeline_type == "coloring-book"

    import time
    last_activity = _last_activity(project_dir)
    now = time.time()

    # Stall detection: an in_progress stage that stopped writing anything.
    for stage_entry in stages:
        if (
            stage_entry["status"] == "in_progress"
            and last_activity
            and (now - last_activity) > STALL_WINDOW_SECONDS
        ):
            stage_entry["stalled"] = True
            stage_entry["stalled_minutes"] = int((now - last_activity) / 60)

    settings = marker.get("settings") or {}
    state: dict[str, Any] = {
        "project_id": project_id,
        "title": marker.get("title") or meta_json.get("name") or project_id.replace("-", " ").title(),
        "pipeline": pipeline_meta,
        "style": marker.get("style"),
        "difficulty": settings.get("difficulty"),
        "commercial_safe": settings.get("commercial_safe"),
        "book_id": marker.get("book_id"),
        "created_at": marker.get("created_at"),
        "has_marker": bool(marker),
        "has_pipeline_state": bool(checkpoints),
        "stages": stages,
        "artifacts": artifacts,
        "steps": [] if is_book else _build_steps(project_dir, artifacts, events),
        "book": _build_book(project_dir, artifacts) if is_book else None,
        "qa": artifacts.get("qa_report"),
        "images": _scan_images(project_dir),
        "events": events,
        "last_activity": last_activity,
        "live": bool(last_activity and (now - last_activity) < LIVE_WINDOW_SECONDS),
    }
    state["poster"] = _find_poster(project_dir, state)
    return state


def summarize_project(project_dir: Path) -> dict[str, Any]:
    """Cheap library-card summary (no full artifact parse of big files)."""
    state = load_board_state(project_dir)
    active = next((s for s in state["stages"] if s["status"] in ("in_progress", "awaiting_human")), None)
    done = [s for s in state["stages"] if s["status"] == "completed"]
    return {
        "project_id": state["project_id"],
        "title": state["title"],
        "pipeline_type": state["pipeline"]["pipeline_type"],
        "has_pipeline_state": state["has_pipeline_state"],
        "poster": state["poster"],
        "live": state["live"],
        "last_activity": state["last_activity"],
        "active_stage": active["name"] if active else None,
        "awaiting_human": bool(active and active["status"] == "awaiting_human"),
        "stage_states": [
            {"name": s["name"], "status": s["status"]}
            for s in state["stages"] if not s.get("undeclared")
        ],
        "completed_count": len(done),
        "difficulty": state["difficulty"],
        "page_count": len((state["book"] or {}).get("pages", [])),
        "qa_status": (state["qa"] or {}).get("status"),
        "book_id": state["book_id"],
    }


def list_projects(projects_dir: Optional[Path] = None) -> list[dict[str, Any]]:
    """Library view: every project directory, live-first then recency."""
    root = Path(projects_dir) if projects_dir else PROJECTS_DIR
    if not root.is_dir():
        return []
    summaries = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith(("_", ".")):
            continue
        try:
            summaries.append(summarize_project(entry))
        except Exception:
            summaries.append({
                "project_id": entry.name,
                "title": entry.name.replace("-", " ").title(),
                "pipeline_type": "unknown",
                "has_pipeline_state": False,
                "poster": None,
                "live": False,
                "last_activity": 0,
                "active_stage": None,
                "awaiting_human": False,
                "stage_states": [],
                "completed_count": 0,
                "difficulty": None,
                "page_count": 0,
                "qa_status": None,
                "book_id": None,
                "error": "unreadable",
            })
    summaries.sort(key=lambda s: (not s["live"], -(s["last_activity"] or 0)))
    return summaries
