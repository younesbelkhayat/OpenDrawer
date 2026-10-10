"""Base class shared by every cartoonize backend.

A backend only implements how ONE crop becomes a cartoon (``generate_job``),
plus optional ``prepare``/``teardown`` hooks to load and free models. This
class owns the rest of the contract so all backends behave the same:
reading the photo_analysis, splitting into jobs, seeds, compositing on white
aligned with the photo, scoring, and returning a valid ``cartoon_set``.
"""

from __future__ import annotations

import json
import random
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lib.imaging import disk, save_rgb
from lib.model_registry import commercial_safe_mode
from styles.style_loader import build_prompt, load_style
from tools.base_tool import BaseTool, Determinism, ToolResult, ToolTier
from tools.cartoon._common import (
    Job,
    compose_jobs,
    crop,
    load_analysis_images,
    plan_jobs,
    score_candidate,
    white_outside,
)

CARTOONIZE_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["output_dir"],
    "properties": {
        "photo_analysis": {"type": "object", "description": "photo_analysis artifact (inline)"},
        "photo_analysis_path": {"type": "string", "description": "…or a path to its JSON file"},
        "output_dir": {"type": "string", "description": "Where candidate PNGs are written"},
        "style": {"type": "string", "default": "livre-jeunesse"},
        "candidates": {"type": "integer", "default": 3, "minimum": 1, "maximum": 8},
        "seed": {"type": "integer", "description": "Base seed; candidate i uses seed + i"},
        "family": {"type": "string", "enum": ["auto", "sdxl", "sd15"], "default": "auto"},
        "identity": {
            "type": "string",
            "enum": ["auto", "faceid", "plus_face", "none"],
            "default": "auto",
            "description": "Face resemblance conditioning (faceid is non-commercial)",
        },
        "commercial_safe": {"type": "boolean", "description": "Defaults to OPENDRAWER_COMMERCIAL_SAFE"},
        "per_subject": {"type": "boolean", "description": "Force one generation per subject"},
        "descriptions": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": "subject id → English description (hair, glasses, clothes) to sharpen the prompt",
        },
        "steps": {"type": "integer"},
        "strength": {"type": "number"},
        "guidance_scale": {"type": "number"},
        "controlnet_scale": {"type": "number"},
        "ip_adapter_scale": {"type": "number"},
    },
}


def load_photo_analysis(inputs: dict[str, Any]) -> dict[str, Any]:
    if isinstance(inputs.get("photo_analysis"), dict):
        return inputs["photo_analysis"]
    if inputs.get("photo_analysis_path"):
        with open(inputs["photo_analysis_path"], encoding="utf-8") as f:
            return json.load(f)
    raise ValueError("photo_analysis or photo_analysis_path is required")


class CartoonizeProvider(BaseTool):
    """Abstract cartoonize backend (not registered: it has an abstract method)."""

    tier = ToolTier.GENERATE
    capability = "cartoonize"
    determinism = Determinism.SEEDED
    input_schema = CARTOONIZE_INPUT_SCHEMA
    side_effects = ["writes candidate PNGs to output_dir"]
    user_visible_verification = [
        "Ressemblance avec la photo",
        "Style livre jeunesse (contours nets, aplats)",
        "Fond blanc, pas de membres en trop",
    ]

    # ---- hooks -----------------------------------------------------------

    def prepare(self, inputs: dict[str, Any], analysis: dict[str, Any], jobs: list[Job]) -> dict[str, Any]:
        """Load models once per execute(). Returns a context passed to generate_job."""
        return {}

    def teardown(self, ctx: dict[str, Any]) -> None:
        """Free models (VRAM) after the run."""

    @abstractmethod
    def generate_job(
        self, job: Job, crop_rgb: np.ndarray, seed: int, ctx: dict[str, Any]
    ) -> np.ndarray:
        """Return an RGB cartoon for one crop (any size; it is resized back)."""

    def describe(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Backend facts recorded in the cartoon_set (family, identity mode...)."""
        return {}

    # ---- shared helpers for subclasses -------------------------------------

    @staticmethod
    def style_params(inputs: dict[str, Any], style: dict[str, Any], family: str) -> dict[str, Any]:
        params = dict((style.get("generation") or {}).get(family) or {})
        for key in ("steps", "strength", "guidance_scale", "controlnet_scale", "ip_adapter_scale"):
            if inputs.get(key) is not None:
                params[key] = inputs[key]
        return params

    @staticmethod
    def prompts_for(style: dict[str, Any], job: Job, descriptions: dict[str, str]) -> tuple[str, str]:
        text = None
        if len(job.subject_ids) == 1:
            text = descriptions.get(job.subject_ids[0])
        return build_prompt(style, job.kind, text or job.description)

    # ---- the shared contract ---------------------------------------------

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        started = time.monotonic()
        try:
            analysis = load_photo_analysis(inputs)
            cutout, fg_mask = load_analysis_images(analysis)
            style_name = inputs.get("style") or "livre-jeunesse"
            style = load_style(style_name)
        except Exception as exc:
            return ToolResult(success=False, error=f"{self.name}: invalid inputs: {exc}")

        if inputs.get("commercial_safe") is None:
            inputs = {**inputs, "commercial_safe": commercial_safe_mode()}
        out_dir = Path(inputs["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        jobs = plan_jobs(analysis, fg_mask, inputs.get("per_subject"))
        n = int(inputs.get("candidates", 3))
        base_seed = inputs.get("seed")
        if base_seed is None:
            base_seed = random.randint(0, 2**31 - 1)

        # The cartoon may slightly overflow the photo silhouette (rounder
        # shapes, bigger head): allow a margin before forcing white.
        h, w = fg_mask.shape
        margin = max(2, int(max(h, w) * 0.015))
        allowed = cv2.dilate(fg_mask.astype(np.uint8), disk(margin)) > 0

        try:
            ctx = self.prepare(inputs, analysis, jobs)
        except Exception as exc:
            return ToolResult(success=False, error=f"{self.name}: could not load backend: {exc}")
        ctx.setdefault("style", style)
        ctx.setdefault("inputs", inputs)
        ctx.setdefault("descriptions", inputs.get("descriptions") or {})

        candidates = []
        try:
            for i in range(n):
                seed = int(base_seed) + i
                results = [(job, self.generate_job(job, crop(cutout, job.box), seed, ctx)) for job in jobs]
                image = white_outside(compose_jobs((h, w), results), allowed)
                path = save_rgb(image, out_dir / f"candidate_{i + 1}.png")
                candidates.append(
                    {
                        "id": f"c{i + 1}",
                        "path": str(path),
                        "seed": seed,
                        "score": score_candidate(image, cutout, fg_mask),
                    }
                )
        except Exception as exc:
            self.teardown(ctx)
            return ToolResult(
                success=False,
                error=f"{self.name}: generation failed after {len(candidates)} candidate(s): {exc}",
                data={"partial_candidates": candidates},
            )
        facts = self.describe(ctx)
        self.teardown(ctx)

        best = max(candidates, key=lambda c: c["score"]["total"])
        first_job = jobs[0]
        prompt, negative = self.prompts_for(style, first_job, ctx["descriptions"])
        cartoon_set = {
            "version": "1.0",
            "backend": self.name,
            "model_family": facts.get("model_family"),
            "style": style_name,
            "identity_mode": facts.get("identity_mode", "none"),
            "commercial_safe": bool(inputs.get("commercial_safe")),
            "memory_strategy": facts.get("memory_strategy"),
            "prompt": prompt,
            "negative_prompt": negative,
            "candidates": candidates,
            "selected_id": best["id"],
            "selection_reason": "auto: highest objective score (structure/color/style); to be confirmed by review",
        }
        return ToolResult(
            success=True,
            data={"cartoon_set": cartoon_set, "jobs": len(jobs), "warnings": facts.get("warnings", [])},
            artifacts=[c["path"] for c in candidates],
            seed=int(base_seed),
            model=facts.get("model"),
            duration_seconds=time.monotonic() - started,
        )
