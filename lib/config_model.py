"""Runtime configuration model for OpenDrawer.

Loads config.yaml and provides typed access. CLI flags override these values
for a single run; nothing here is required (defaults are sensible).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class CheckpointPolicy(str, Enum):
    GUIDED = "guided"
    MANUAL_ALL = "manual_all"
    AUTO_NONCREATIVE = "auto_noncreative"


class CheckpointConfig(BaseModel):
    policy: CheckpointPolicy = CheckpointPolicy.GUIDED


class GenerationConfig(BaseModel):
    backend: str = "auto"  # auto | diffusers | comfyui | opencv
    family: str = "auto"  # auto | sdxl | sd15
    style: str = "livre-jeunesse"
    candidates: int = 3
    identity: str = "auto"  # auto | faceid | plus_face | none
    seed: Optional[int] = None


class ColoringConfig(BaseModel):
    difficulty: str = "medium"
    color_source: str = "photo"  # photo | cartoon


class PageConfig(BaseModel):
    paper: str = "A4"
    dpi: int = 300
    margin_mm: float = 10.0
    thumbnail_ratio: float = 0.26


class LicensingConfig(BaseModel):
    commercial_safe: bool = False


class OpenDrawerConfig(BaseModel):
    """Top-level runtime configuration."""

    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    coloring: ColoringConfig = Field(default_factory=ColoringConfig)
    page: PageConfig = Field(default_factory=PageConfig)
    licensing: LicensingConfig = Field(default_factory=LicensingConfig)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "OpenDrawerConfig":
        """Load config from YAML file. Falls back to defaults if file missing."""
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent / "config.yaml"

        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            return cls.model_validate(raw)

        return cls()
