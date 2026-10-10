"""Cartoonize selector: routes to the best available backend.

``backend="auto"`` tries diffusers → comfyui → opencv in that order and
reports which one ran. Agents must announce the chosen backend before the
run and must not switch backends after a failure without user approval
(AGENT_GUIDE.md › Decision Communication Contract): pass an explicit
``backend`` to pin it.
"""

from __future__ import annotations

from typing import Any

from tools.base_tool import BaseTool, Determinism, ToolResult, ToolRuntime, ToolStability, ToolStatus, ToolTier
from tools.cartoon._provider import CARTOONIZE_INPUT_SCHEMA

BACKENDS = {
    "diffusers": "diffusers_cartoonize",
    "comfyui": "comfyui_cartoonize",
    "opencv": "opencv_cartoonize",
}
AUTO_ORDER = ("diffusers", "comfyui", "opencv")


class Cartoonize(BaseTool):
    name = "cartoonize"
    version = "1.0.0"
    tier = ToolTier.GENERATE
    capability = "cartoonize"
    provider = "selector"
    stability = ToolStability.BETA
    determinism = Determinism.SEEDED
    runtime = ToolRuntime.HYBRID
    capabilities = ["photo_to_cartoon"]
    best_for = ["point d'entrée unique pour l'étape cartoon"]
    fallback_tools = list(BACKENDS.values())

    input_schema = {
        **CARTOONIZE_INPUT_SCHEMA,
        "properties": {
            **CARTOONIZE_INPUT_SCHEMA["properties"],
            "backend": {"type": "string", "enum": ["auto", *BACKENDS], "default": "auto"},
        },
    }

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE  # opencv backend is always there

    def resolve_backend(self, requested: str = "auto") -> str:
        from tools.tool_registry import registry

        registry.ensure_discovered()
        if requested != "auto":
            if requested not in BACKENDS:
                raise ValueError(f"Unknown backend {requested!r}; use one of {list(BACKENDS)}")
            return requested
        for backend in AUTO_ORDER:
            tool = registry.get(BACKENDS[backend])
            if tool is not None and tool.get_status() == ToolStatus.AVAILABLE:
                return backend
        return "opencv"

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        from tools.tool_registry import registry

        tool = registry.get(BACKENDS[self.resolve_backend(inputs.get("backend", "auto"))])
        return tool.estimate_runtime(inputs) if tool else 0.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        from tools.tool_registry import registry

        try:
            backend = self.resolve_backend(inputs.get("backend", "auto"))
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))
        tool = registry.get(BACKENDS[backend])
        if tool is None:
            return ToolResult(success=False, error=f"Backend tool {BACKENDS[backend]} is not registered")
        result = tool.execute({k: v for k, v in inputs.items() if k != "backend"})
        result.data.setdefault("backend", backend)
        if inputs.get("backend", "auto") == "auto" and backend == "opencv":
            result.data.setdefault("warnings", []).append(
                "No diffusion backend available: used the OpenCV fallback (lower quality). "
                "Run `make install-gpu` and `python -m opendrawer models download sdxl`."
            )
        return result
