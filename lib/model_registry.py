"""Model catalog: what weights exist, where they live, and whether a run may use them.

``models.yaml`` is the single source of truth for every set of weights a tool
can load. Tools declare the keys they need in ``required_models`` and resolve
local paths through this registry, which also enforces ``--commercial-safe``.

Downloads only ever happen through ``download()`` (CLI: ``python -m opendrawer
models download``). Tools never fetch weights on their own, so a run is fully
offline once the models are on disk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from lib.paths import MODELS_DIR, REPO_ROOT

CATALOG_PATH = REPO_ROOT / "models.yaml"
COMPLETE_MARKER = ".opendrawer-complete"


class LicenseError(PermissionError):
    """A restricted model was requested in commercial-safe mode."""


class ModelNotDownloaded(FileNotFoundError):
    """Weights are missing on disk; the message says how to fetch them."""


@dataclass
class ModelSpec:
    key: str
    kind: str
    license: str
    commercial_safe: bool
    role: str = ""
    family: Optional[str] = None
    repo: Optional[str] = None
    url: Optional[str] = None
    filename: Optional[str] = None
    manager: Optional[str] = None
    weight_name: Optional[str] = None
    subfolder: Optional[str] = None
    allow_patterns: list[str] = field(default_factory=list)
    approx_size_mb: int = 0
    notes: str = ""

    @classmethod
    def from_dict(cls, key: str, raw: dict[str, Any]) -> "ModelSpec":
        known = {f for f in cls.__dataclass_fields__ if f != "key"}
        return cls(key=key, **{k: v for k, v in raw.items() if k in known})


def commercial_safe_mode() -> bool:
    """True when the current process runs with ``--commercial-safe``."""
    return os.environ.get("OPENDRAWER_COMMERCIAL_SAFE", "").strip().lower() in {"1", "true", "yes"}


class ModelRegistry:
    def __init__(self, catalog_path: Path = CATALOG_PATH, models_dir: Optional[Path] = None) -> None:
        self.catalog_path = catalog_path
        self._models_dir = models_dir
        self._specs: Optional[dict[str, ModelSpec]] = None
        self._presets: dict[str, list[str]] = {}

    # ------------------------------------------------------------------ catalog

    @property
    def models_dir(self) -> Path:
        return self._models_dir or MODELS_DIR

    def _load(self) -> dict[str, ModelSpec]:
        if self._specs is None:
            with open(self.catalog_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            self._specs = {
                key: ModelSpec.from_dict(key, spec) for key, spec in (raw.get("models") or {}).items()
            }
            self._presets = {k: list(v) for k, v in (raw.get("presets") or {}).items()}
        return self._specs

    def keys(self) -> list[str]:
        return list(self._load())

    def presets(self) -> dict[str, list[str]]:
        self._load()
        return dict(self._presets)

    def spec(self, key: str) -> ModelSpec:
        specs = self._load()
        if key not in specs:
            raise KeyError(f"Unknown model {key!r}. Known models: {sorted(specs)}")
        return specs[key]

    def expand(self, names: Iterable[str]) -> list[str]:
        """Resolve a mix of model keys and preset names into model keys."""
        self._load()
        out: list[str] = []
        for name in names:
            members = self._presets.get(name, [name])
            for key in members:
                self.spec(key)
                if key not in out:
                    out.append(key)
        return out

    # ------------------------------------------------------------------ licensing

    def check_allowed(self, key: str, commercial_safe: Optional[bool] = None) -> None:
        if commercial_safe is None:
            commercial_safe = commercial_safe_mode()
        spec = self.spec(key)
        if commercial_safe and not spec.commercial_safe:
            raise LicenseError(
                f"Model {key!r} ({spec.license}) is not allowed with --commercial-safe."
            )

    # ------------------------------------------------------------------ disk

    def local_dir(self, key: str) -> Path:
        return self.models_dir / key

    def is_downloaded(self, key: str) -> bool:
        spec = self.spec(key)
        if spec.kind == "external":
            # Managed by its own library cache (rembg, insightface): we cannot
            # tell reliably, and the library downloads on first use.
            return True
        local = self.local_dir(key)
        if spec.kind == "url":
            return (local / (spec.filename or "")).is_file()
        return (local / COMPLETE_MARKER).is_file()

    def path(self, key: str, commercial_safe: Optional[bool] = None) -> Path:
        """Local directory (or file for ``url`` models), after licence + presence checks."""
        self.check_allowed(key, commercial_safe)
        spec = self.spec(key)
        if not self.is_downloaded(key):
            raise ModelNotDownloaded(
                f"Model {key!r} is not downloaded. Run: python -m opendrawer models download {key}"
            )
        local = self.local_dir(key)
        if spec.kind == "url":
            return local / (spec.filename or "")
        return local

    def status(self, keys: Optional[Iterable[str]] = None) -> list[dict[str, Any]]:
        rows = []
        for key in keys or self.keys():
            spec = self.spec(key)
            rows.append(
                {
                    "key": key,
                    "kind": spec.kind,
                    "role": spec.role,
                    "family": spec.family,
                    "license": spec.license,
                    "commercial_safe": spec.commercial_safe,
                    "downloaded": self.is_downloaded(key),
                    "approx_size_mb": spec.approx_size_mb,
                }
            )
        return rows

    # ------------------------------------------------------------------ download

    def download(self, key: str, *, commercial_safe: Optional[bool] = None) -> Path:
        """Fetch one model. Network is used here and nowhere else."""
        self.check_allowed(key, commercial_safe)
        spec = self.spec(key)
        local = self.local_dir(key)
        if spec.kind == "external":
            return local
        local.mkdir(parents=True, exist_ok=True)
        if spec.kind == "url":
            import requests

            target = local / (spec.filename or Path(spec.url or "").name)
            tmp = target.with_suffix(target.suffix + ".part")
            with requests.get(spec.url, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            os.replace(tmp, target)
            return target
        if spec.kind == "huggingface":
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:  # pragma: no cover - depends on env
                raise RuntimeError("pip install huggingface_hub to download models") from exc
            snapshot_download(
                repo_id=spec.repo,
                local_dir=str(local),
                allow_patterns=spec.allow_patterns or None,
            )
            (local / COMPLETE_MARKER).write_text(spec.repo or key, encoding="utf-8")
            return local
        raise ValueError(f"Unsupported model kind {spec.kind!r} for {key!r}")


model_registry = ModelRegistry()
