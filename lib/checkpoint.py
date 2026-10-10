"""Checkpoint writer/reader for pipeline state persistence.

Each stage writes a checkpoint after completion. The orchestrator uses
checkpoints to resume pipelines and to present state at human checkpoints.
"""

from __future__ import annotations

import json
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import jsonschema

from schemas.artifacts import ARTIFACT_NAMES, validate_artifact

def get_pipeline_stages(pipeline_type: str | None) -> list[str]:
    """Return the ordered stage list for a specific pipeline (empty if unknown)."""
    if not pipeline_type:
        return []
    try:
        from lib.pipeline_loader import load_pipeline_readonly, get_stage_order
        manifest = load_pipeline_readonly(pipeline_type)
        return get_stage_order(manifest)
    except Exception:
        return []


def all_known_stages() -> set[str]:
    """Union of stage names across every pipeline manifest."""
    from lib.pipeline_loader import list_pipelines
    stages: set[str] = set()
    for name in list_pipelines():
        stages.update(get_pipeline_stages(name))
    return stages


def get_stage_required_artifacts(pipeline_type: str | None, stage: str) -> list[str]:
    """Artifacts a stage must carry when completed (its manifest ``produces``)."""
    if not pipeline_type:
        return []
    try:
        from lib.pipeline_loader import load_pipeline_readonly, get_stage_produces
        return get_stage_produces(load_pipeline_readonly(pipeline_type), stage)
    except Exception:
        return []


CHECKPOINT_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / "schemas"
    / "checkpoints"
    / "checkpoint.schema.json"
)

# Canonical project root. Checkpoints, artifacts, and the project marker all
# live under PROJECTS_DIR/<project_id>/ — this is the location the Backlot
# board watches. Callers may still pass a different pipeline_dir (tests do),
# but production runs should use the default.
from lib.paths import PROJECTS_DIR  # noqa: E402  (single source of truth)

PROJECT_MARKER_FILENAME = "project.json"
HISTORY_DIRNAME = "history"

# Canonical layout of a project directory (see docs/ARCHITECTURE.md).
PROJECT_SUBDIRS = (
    "artifacts",
    "input",
    "assets/intake",
    "assets/cartoon",
    "assets/line_art",
    "assets/regions",
    "output",
)


class CheckpointValidationError(ValueError):
    """Raised when a checkpoint or its canonical artifacts are invalid."""


def _validate_style(style: str | None) -> None:
    """Fail closed when a checkpoint names a cartoon style that cannot load."""

    if style is None:
        return
    try:
        from styles.style_loader import list_styles, load_style

        load_style(style)
    except Exception as exc:
        try:
            available = list_styles()
        except Exception:
            available = []
        raise CheckpointValidationError(
            f"Unknown or invalid style {style!r}. "
            f"Available styles: {available}. Underlying error: {exc}"
        ) from exc


@lru_cache(maxsize=1)
def _load_checkpoint_schema() -> dict[str, Any]:
    with open(CHECKPOINT_SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


def _validate_artifacts_for_stage(
    pipeline_type: str | None,
    stage: str,
    status: str,
    artifacts: dict[str, Any],
) -> None:
    # Required artifacts come from the stage's ``produces`` list in the
    # pipeline manifest — the manifest is the single source of truth.
    if status in {"completed", "awaiting_human"}:
        missing = [
            name for name in get_stage_required_artifacts(pipeline_type, stage)
            if name not in artifacts
        ]
        if missing:
            raise CheckpointValidationError(
                f"Stage {stage!r} with status {status!r} must include "
                f"artifacts {missing}"
            )

    for artifact_name, artifact_data in artifacts.items():
        if artifact_name not in ARTIFACT_NAMES:
            continue
        if not isinstance(artifact_data, dict):
            raise CheckpointValidationError(
                f"Artifact {artifact_name!r} must be a JSON object matching its schema"
            )
        try:
            validate_artifact(artifact_name, artifact_data)
        except Exception as exc:
            raise CheckpointValidationError(
                f"Artifact {artifact_name!r} failed schema validation: {exc}"
            ) from exc


def validate_checkpoint(checkpoint: dict[str, Any]) -> None:
    """Validate checkpoint structure and canonical artifact payloads.

    Uses pipeline_type (if present) to resolve the valid stage list.
    Falls back to the union of all manifests' stages when it is absent.
    """
    stage = checkpoint.get("stage")
    status = checkpoint.get("status")
    artifacts = checkpoint.get("artifacts")
    pipeline_type = checkpoint.get("pipeline_type")

    valid_stages = set(get_pipeline_stages(pipeline_type)) or all_known_stages()

    if not isinstance(stage, str) or stage not in valid_stages:
        raise CheckpointValidationError(
            f"Invalid stage: {stage!r} for pipeline {pipeline_type!r}. "
            f"Valid stages: {sorted(valid_stages)}"
        )
    if not isinstance(status, str):
        raise CheckpointValidationError(f"Invalid status: {status!r}")
    if not isinstance(artifacts, dict):
        raise CheckpointValidationError("Checkpoint artifacts must be a dictionary")

    _validate_artifacts_for_stage(pipeline_type, stage, status, artifacts)

    try:
        jsonschema.validate(instance=checkpoint, schema=_load_checkpoint_schema())
    except jsonschema.ValidationError as exc:
        raise CheckpointValidationError(f"Checkpoint failed schema validation: {exc.message}") from exc


def _checkpoint_path(pipeline_dir: Path, project_id: str, stage: str) -> Path:
    return pipeline_dir / project_id / f"checkpoint_{stage}.json"


def init_project(
    project_id: str,
    *,
    title: str,
    pipeline_type: str,
    pipeline_dir: Optional[Path] = None,
    style: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    """Initialize a project workspace with the canonical layout + marker file.

    Creates projects/<project_id>/ with the standard subdirectories and writes
    project.json — the marker the Backlot board uses to render a project's
    identity and stage rail before the first checkpoint exists.

    Idempotent: re-running preserves the original created_at and merges fields.
    ``extra`` merges run settings (difficulty, commercial_safe, book_id...)
    into the marker. Returns the project directory.
    """
    _validate_style(style)
    base = pipeline_dir or PROJECTS_DIR
    project_dir = base / project_id
    for sub in PROJECT_SUBDIRS:
        (project_dir / sub).mkdir(parents=True, exist_ok=True)

    marker_path = project_dir / PROJECT_MARKER_FILENAME
    marker: dict[str, Any] = {}
    if marker_path.exists():
        try:
            with open(marker_path, encoding="utf-8") as f:
                marker = json.load(f)
        except (json.JSONDecodeError, OSError):
            marker = {}

    marker.setdefault("version", "1.0")
    marker.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    marker["project_id"] = project_id
    marker["title"] = title
    marker["pipeline_type"] = pipeline_type
    if style is not None:
        marker["style"] = style
    if extra:
        marker.update(extra)

    with open(marker_path, "w", encoding="utf-8") as f:
        json.dump(marker, f, indent=2)

    return project_dir


def _stage_requires_approval(pipeline_type: Optional[str], stage: str) -> Optional[bool]:
    """Read human_approval_default for a stage from its pipeline manifest.

    Returns None when the stage isn't declared in the manifest or no
    pipeline_type was given — the caller then falls back to the value the
    agent passed in.

    A *provided but unknown* pipeline_type raises: a typo must not silently
    disable gate enforcement (fail-closed, not fail-open). Other manifest
    load failures are logged and fall back — a corrupt manifest shouldn't
    strand an otherwise-valid run, but the degradation must be visible.
    """
    if not pipeline_type or pipeline_type == "unknown":
        return None
    from lib.pipeline_loader import get_stage_human_approval_default, load_pipeline_readonly
    try:
        manifest = load_pipeline_readonly(pipeline_type)
    except FileNotFoundError:
        raise CheckpointValidationError(
            f"Unknown pipeline_type {pipeline_type!r} — cannot resolve gate "
            f"policy for stage {stage!r}. Check the spelling against "
            f"pipeline_defs/*.yaml."
        )
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "Gate policy unavailable for pipeline %r (%s) — falling back to "
            "the caller's human_approval_required flag.", pipeline_type, exc,
        )
        return None
    return get_stage_human_approval_default(manifest, stage)


def _enforce_stage_prerequisites(
    pipeline_dir: Path,
    project_id: str,
    pipeline_type: str | None,
    stage: str,
    status: str,
) -> None:
    """Require completed, approved predecessors before advancing a stage.

    ``in_progress`` and failure heartbeats remain writable so an operator can
    inspect or resume a broken run. Only lifecycle advancement
    (``awaiting_human``/``completed``) is gated.
    """

    if status not in {"awaiting_human", "completed"}:
        return
    if not pipeline_type or pipeline_type == "unknown":
        return

    stages = get_pipeline_stages(pipeline_type)
    if stage not in stages:
        return

    incomplete: list[str] = []
    unapproved: list[str] = []
    for predecessor in stages[: stages.index(stage)]:
        path = _checkpoint_path(pipeline_dir, project_id, predecessor)
        if not path.exists():
            incomplete.append(predecessor)
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                checkpoint = json.load(handle)
            validate_checkpoint(checkpoint)
        except (OSError, json.JSONDecodeError, CheckpointValidationError):
            incomplete.append(predecessor)
            continue
        if (
            checkpoint.get("project_id") != project_id
            or checkpoint.get("pipeline_type") != pipeline_type
            or checkpoint.get("stage") != predecessor
        ):
            incomplete.append(predecessor)
            continue
        if checkpoint.get("status") != "completed":
            incomplete.append(predecessor)
            continue
        if _stage_requires_approval(pipeline_type, predecessor) and not checkpoint.get(
            "human_approved"
        ):
            unapproved.append(predecessor)

    if incomplete or unapproved:
        details = []
        if incomplete:
            details.append(f"incomplete or missing: {incomplete}")
        if unapproved:
            details.append(f"completed without required approval: {unapproved}")
        raise CheckpointValidationError(
            f"PREREQUISITE VIOLATION: stage {stage!r} cannot advance; "
            + "; ".join(details)
            + f". Pipeline order: {stages}."
        )


def _archive_superseded_checkpoint(path: Path, stage: str) -> None:
    """Copy an existing checkpoint into history/ before it is overwritten.

    Preserves the full run record: stage re-runs (script v1 → v2) and gate
    transitions (awaiting_human → completed) remain reconstructable. Repeated
    in_progress refreshes are NOT archived — they are partial-progress
    heartbeats, not versions.

    Archiving is best-effort and must never crash a checkpoint write: the
    Backlot watcher may hold the file open (Windows denies renames of open
    files), so we copy rather than move, and swallow archival I/O failures.
    """
    if not path.exists():
        return
    try:
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
    except (json.JSONDecodeError, OSError):
        existing = {}
    if existing.get("status") == "in_progress":
        return

    try:
        import shutil
        stamp = str(existing.get("timestamp", ""))
        safe_stamp = "".join(c for c in stamp if c.isalnum()) or f"{path.stat().st_mtime_ns}"
        history_dir = path.parent / HISTORY_DIRNAME
        history_dir.mkdir(parents=True, exist_ok=True)
        target = history_dir / f"checkpoint_{stage}_{safe_stamp}.json"
        if target.exists():
            target = history_dir / f"checkpoint_{stage}_{safe_stamp}_{path.stat().st_mtime_ns}.json"
        shutil.copyfile(path, target)
    except OSError:
        import logging
        logging.getLogger(__name__).warning(
            "Could not archive superseded checkpoint %s to history/", path
        )


def _decision_log_path(pipeline_dir: Path, project_id: str) -> Path:
    return pipeline_dir / project_id / "decision_log.json"


def _merge_decision_log(
    pipeline_dir: Path, project_id: str, new_log: dict[str, Any]
) -> None:
    """Append new decisions to the project-level decision log.

    Each stage may produce decisions. This function merges them into a
    single cumulative file so reviewers and the bench can inspect the
    full audit trail.
    """
    path = _decision_log_path(pipeline_dir, project_id)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = {
            "version": "1.0",
            "project_id": project_id,
            "decisions": [],
        }

    existing_ids = {d["decision_id"] for d in existing.get("decisions", [])}
    for decision in new_log.get("decisions", []):
        if decision.get("decision_id") not in existing_ids:
            existing["decisions"].append(decision)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)


def write_checkpoint(
    pipeline_dir: Path,
    project_id: str,
    stage: str,
    status: str,
    artifacts: dict[str, Any],
    *,
    pipeline_type: Optional[str] = None,
    style: Optional[str] = None,
    checkpoint_policy: str = "guided",
    human_approval_required: bool = False,
    human_approved: bool = False,
    review: Optional[dict] = None,
    error: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Path:
    """Write a checkpoint file for a pipeline stage."""
    # Backfill identity fields from the project marker so omitted kwargs
    # cannot bypass either gate enforcement or style validation.
    marker = None
    marker_path = pipeline_dir / project_id / PROJECT_MARKER_FILENAME
    if marker_path.exists() and (not pipeline_type or not style):
        try:
            with open(marker_path, encoding="utf-8") as f:
                marker = json.load(f)
        except (json.JSONDecodeError, OSError):
            marker = None
    if isinstance(marker, dict):
        if not pipeline_type and marker.get("pipeline_type"):
            pipeline_type = marker["pipeline_type"]
        if not style and marker.get("style"):
            style = marker["style"]
    _validate_style(style)

    valid_stages = set(get_pipeline_stages(pipeline_type)) or all_known_stages()
    if stage not in valid_stages:
        raise ValueError(
            f"Invalid stage: {stage!r} for pipeline {pipeline_type!r}. "
            f"Valid stages: {sorted(valid_stages)}"
        )

    # --- Gate enforcement (GI-4) ---
    # The pipeline manifest is the binding source of truth for whether a stage
    # gates on human approval; a caller may gate MORE strictly (e.g. a
    # manual_all checkpoint policy) but never less. A gated stage can only be
    # written "completed" with explicit evidence of approval
    # (human_approved=True). Skipping a gate is a hard error.
    #
    # Enforcement happens at write time only: pre-existing checkpoints written
    # before gating (or by hand) still read as completed — deliberate
    # back-compat so in-flight and legacy projects keep resuming.
    manifest_gate = _stage_requires_approval(pipeline_type, stage)
    gated = bool(manifest_gate) or human_approval_required
    if gated:
        human_approval_required = True
        if status == "completed" and not human_approved:
            gate_source = (
                f"human_approval_default: true in the {pipeline_type!r} manifest"
                if manifest_gate
                else "human_approval_required=True was passed by the caller"
            )
            raise CheckpointValidationError(
                f"GATE VIOLATION: stage {stage!r} requires human approval "
                f"({gate_source}) but status='completed' was written without "
                f"human_approved=True. Correct protocol: write "
                f"status='awaiting_human', present the artifact summary to the "
                f"user, END YOUR TURN, and only after the user approves "
                f"re-write with status='completed', human_approved=True."
            )

    _enforce_stage_prerequisites(
        pipeline_dir,
        project_id,
        pipeline_type,
        stage,
        status,
    )

    checkpoint = {
        "version": "1.0",
        "project_id": project_id,
        "pipeline_type": pipeline_type or "unknown",
        "stage": stage,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint_policy": checkpoint_policy,
        "human_approval_required": human_approval_required,
        "human_approved": human_approved,
        "artifacts": artifacts,
    }
    if style is not None:
        checkpoint["style"] = style
    if review is not None:
        checkpoint["review"] = review
    if error is not None:
        checkpoint["error"] = error
    if metadata is not None:
        checkpoint["metadata"] = metadata

    # Merge decision_log: if this checkpoint carries new decisions, append
    # them to the project-level decision log file (cumulative audit trail).
    if "decision_log" in artifacts and isinstance(artifacts["decision_log"], dict):
        _merge_decision_log(pipeline_dir, project_id, artifacts["decision_log"])

    validate_checkpoint(checkpoint)

    path = _checkpoint_path(pipeline_dir, project_id, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize to a temp file first so a mid-write failure (disk full,
    # unserializable metadata) can never leave the stage with a truncated
    # current checkpoint; then archive the superseded file and swap in the
    # new one atomically.
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)
    # Preserve run history: a superseded completed/awaiting_human checkpoint
    # is copied to history/ (stage versioning, gate audit trail, replay).
    _archive_superseded_checkpoint(path, stage)
    import os
    os.replace(tmp_path, path)

    return path


def read_checkpoint(
    pipeline_dir: Path, project_id: str, stage: str
) -> Optional[dict[str, Any]]:
    """Read a checkpoint file. Returns None if not found."""
    path = _checkpoint_path(pipeline_dir, project_id, stage)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        checkpoint = json.load(f)
    validate_checkpoint(checkpoint)
    return checkpoint


def get_latest_checkpoint(
    pipeline_dir: Path, project_id: str
) -> Optional[dict[str, Any]]:
    """Find the most recent checkpoint for a project (by file mtime)."""
    project_dir = pipeline_dir / project_id
    if not project_dir.exists():
        return None

    checkpoints = sorted(
        project_dir.glob("checkpoint_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not checkpoints:
        return None

    with open(checkpoints[0], encoding="utf-8") as f:
        checkpoint = json.load(f)
    validate_checkpoint(checkpoint)
    return checkpoint


def get_completed_stages(
    pipeline_dir: Path, project_id: str, pipeline_type: str | None = None
) -> list[str]:
    """Return list of stages that have a completed checkpoint.

    When pipeline_type is provided, only checks stages defined in that
    pipeline's manifest — preventing false positives from leftover
    checkpoints of a different pipeline type.
    """
    stages_to_check = get_pipeline_stages(pipeline_type)
    completed = []
    for stage in stages_to_check:
        cp = read_checkpoint(pipeline_dir, project_id, stage)
        if cp and cp.get("status") == "completed":
            completed.append(stage)
    return completed


def get_next_stage(
    pipeline_dir: Path, project_id: str, pipeline_type: str | None = None
) -> Optional[str]:
    """Determine the next stage to run based on completed checkpoints.

    Uses the pipeline-specific stage order. When pipeline_type is omitted it
    is read from the project marker.
    """
    if not pipeline_type:
        marker_path = pipeline_dir / project_id / PROJECT_MARKER_FILENAME
        if marker_path.exists():
            try:
                with open(marker_path, encoding="utf-8") as f:
                    pipeline_type = json.load(f).get("pipeline_type")
            except (OSError, json.JSONDecodeError):
                pipeline_type = None
    stages = get_pipeline_stages(pipeline_type)
    completed = set(get_completed_stages(pipeline_dir, project_id, pipeline_type))
    for stage in stages:
        if stage not in completed:
            return stage
    return None
