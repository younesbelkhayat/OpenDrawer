"""Cartoon style presets (styles/*.yaml).

A style tells the cartoonize tools how to prompt and parameterize a model so
the result reads as a given illustration style. Validated against
``schemas/styles/style.schema.json``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema
import yaml

STYLES_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = STYLES_DIR.parent / "schemas" / "styles" / "style.schema.json"


@lru_cache(maxsize=1)
def _schema() -> dict[str, Any]:
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


def list_styles() -> list[str]:
    return sorted(p.stem for p in STYLES_DIR.glob("*.yaml"))


@lru_cache(maxsize=16)
def _load_cached(name: str) -> dict[str, Any]:
    path = STYLES_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown style {name!r}. Available: {list_styles()}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    jsonschema.validate(instance=data, schema=_schema())
    return data


def load_style(name: str) -> dict[str, Any]:
    """Return a validated style preset (a fresh copy the caller may mutate)."""
    return json.loads(json.dumps(_load_cached(name)))


def build_prompt(style: dict[str, Any], subject_kind: str, subject_text: str | None = None) -> tuple[str, str]:
    """Positive/negative prompt for one subject."""
    prompt = style["prompt"]
    defaults = prompt.get("subject_defaults") or {}
    subject = subject_text or defaults.get(subject_kind) or defaults.get("person") or "a person"
    positive = " ".join(prompt["positive"].split()).replace("{subject}", subject)
    negative = " ".join(prompt.get("negative", "").split())
    return positive, negative
