"""Tool registry with status, stability, and support-envelope reporting.

The registry discovers all registered tools, reports their availability,
and lets the orchestrator/agents query capabilities by tier, status, etc.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from types import ModuleType
from typing import Any, Optional

from tools.base_tool import BaseTool, ToolStatus, ToolTier, ToolStability


# Unicode punctuation that breaks on Windows cp1252 stdout. Map each to an
# ASCII equivalent. This only touches strings rendered by registry helpers
# that an agent is likely to print to the user at preflight — not docstrings,
# comments, or markdown.
_UNICODE_DASH_REPLACEMENTS = {
    "\u2014": "--",   # em dash
    "\u2013": "-",    # en dash
    "\u2212": "-",    # minus sign
    "\u2018": "'",    # left single quote
    "\u2019": "'",    # right single quote
    "\u201c": '"',    # left double quote
    "\u201d": '"',    # right double quote
    "\u2026": "...",  # ellipsis
}


def _scrub_unicode_dashes(value: Any) -> Any:
    """Recursively normalize unicode punctuation in str leaves to ASCII.

    Used to keep `provider_menu_summary()` output readable on Windows cp1252
    stdout. Does NOT modify dict/list structure or non-string values.
    """
    if isinstance(value, str):
        out = value
        for needle, repl in _UNICODE_DASH_REPLACEMENTS.items():
            if needle in out:
                out = out.replace(needle, repl)
        return out
    if isinstance(value, list):
        return [_scrub_unicode_dashes(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_scrub_unicode_dashes(item) for item in value)
    if isinstance(value, dict):
        return {k: _scrub_unicode_dashes(v) for k, v in value.items()}
    return value


class ToolRegistry:
    """Central registry of all OpenDrawer tools."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._discovered_packages: set[str] = set()

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        if not tool.name:
            raise ValueError("Tool must have a non-empty name")
        self._tools[tool.name] = tool

    def clear(self) -> None:
        """Clear registered tools and discovery state."""
        self._tools.clear()
        self._discovered_packages.clear()

    def register_module(self, module: ModuleType) -> list[str]:
        """Register all concrete BaseTool subclasses defined in a module."""
        registered: list[str] = []
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls is BaseTool or not issubclass(cls, BaseTool):
                continue
            if cls.__module__ != module.__name__ or inspect.isabstract(cls):
                continue
            tool = cls()
            self.register(tool)
            registered.append(tool.name)
        return registered

    @staticmethod
    def _load_dotenv() -> None:
        """Load .env file into os.environ if present, so tools see local settings."""
        from pathlib import Path
        import os
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if not env_path.is_file():
            return
        import re
        with open(env_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # Quoted value: take the content inside the quotes verbatim.
                if value[:1] in ("'", '"'):
                    quote = value[0]
                    end = value.find(quote, 1)
                    value = value[1:end] if end != -1 else value[1:]
                else:
                    # Strip an inline comment ('#' at line start or after
                    # whitespace) so "KEY=   # note" yields "" not "# note".
                    match = re.search(r"(^|\s)#", value)
                    if match:
                        value = value[: match.start()]
                    value = value.strip()
                if key and key not in os.environ:
                    os.environ[key] = value

    def discover(self, package_name: str = "tools") -> list[str]:
        """Import a package tree and register any concrete tools it defines."""
        self._load_dotenv()
        package = importlib.import_module(package_name)
        discovered: list[str] = []
        package_paths = getattr(package, "__path__", None)
        if package_paths is None:
            return self.register_module(package)

        for module_info in pkgutil.walk_packages(package_paths, f"{package.__name__}."):
            if module_info.name.endswith(".base_tool") or module_info.name.endswith(".tool_registry"):
                continue
            module = importlib.import_module(module_info.name)
            discovered.extend(self.register_module(module))

        self._discovered_packages.add(package_name)
        return discovered

    def ensure_discovered(self, package_name: str = "tools") -> None:
        """Load tool modules once before reporting capabilities."""
        if package_name not in self._discovered_packages:
            self.discover(package_name)

    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_all(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_by_tier(self, tier: ToolTier) -> list[BaseTool]:
        """Get all tools in a given tier."""
        return [t for t in self._tools.values() if t.tier == tier]

    def get_by_capability(self, capability: str) -> list[BaseTool]:
        """Get all tools registered for a top-level capability family."""
        return [t for t in self._tools.values() if t.capability == capability]

    def get_by_provider(self, provider: str) -> list[BaseTool]:
        """Get all tools backed by a specific provider."""
        return [t for t in self._tools.values() if t.provider == provider]

    def get_by_status(self, status: ToolStatus) -> list[BaseTool]:
        """Get all tools with a given status."""
        return [t for t in self._tools.values() if t.get_status() == status]

    def get_available(self) -> list[BaseTool]:
        """Get all tools that are currently available."""
        return self.get_by_status(ToolStatus.AVAILABLE)

    def get_unavailable(self) -> list[BaseTool]:
        """Get all tools that are currently unavailable."""
        return self.get_by_status(ToolStatus.UNAVAILABLE)

    def get_by_stability(self, stability: ToolStability) -> list[BaseTool]:
        """Get all tools at a given stability level."""
        return [t for t in self._tools.values() if t.stability == stability]

    def find_by_capability(self, capability: str) -> list[BaseTool]:
        """Find tools that declare a given capability."""
        return [
            t for t in self._tools.values()
            if capability in t.capabilities
        ]

    def find_fallback(self, tool_name: str) -> Optional[BaseTool]:
        """Find the fallback tool for a given tool, if declared and available."""
        tool = self.get(tool_name)
        if tool is None:
            return None
        candidates = list(tool.fallback_tools or [])
        if tool.fallback and tool.fallback not in candidates:
            candidates.append(tool.fallback)
        for name in candidates:
            fb = self.get(name)
            if fb and fb.get_status() == ToolStatus.AVAILABLE:
                return fb
        return None

    def support_envelope(self) -> dict[str, Any]:
        """Generate a full support-envelope report for all tools.

        Returns a dict mapping tool name to its contract info + live status.
        This is the primary report the orchestrator uses to understand
        what the system can and cannot do.
        """
        self.ensure_discovered()
        report: dict[str, Any] = {}
        for name, tool in self._tools.items():
            info = tool.get_info()
            report[name] = info
        return report

    def capability_catalog(self) -> dict[str, list[dict[str, Any]]]:
        """Group the support envelope by top-level capability."""
        self.ensure_discovered()
        grouped: dict[str, list[dict[str, Any]]] = {}
        for tool in self._tools.values():
            grouped.setdefault(tool.capability, []).append(tool.get_info())
        for items in grouped.values():
            items.sort(key=lambda item: (item["provider"], item["name"]))
        return dict(sorted(grouped.items()))

    def provider_catalog(self) -> dict[str, list[dict[str, Any]]]:
        """Group the support envelope by provider."""
        self.ensure_discovered()
        grouped: dict[str, list[dict[str, Any]]] = {}
        for tool in self._tools.values():
            grouped.setdefault(tool.provider, []).append(tool.get_info())
        for items in grouped.values():
            items.sort(key=lambda item: (item["capability"], item["name"]))
        return dict(sorted(grouped.items()))

    def tier_summary(self) -> dict[str, dict[str, int]]:
        """Summarize tool counts by tier and status.

        Returns:
            {"core": {"available": 5, "unavailable": 2, "degraded": 0}, ...}
        """
        summary: dict[str, dict[str, int]] = {}
        for tier in ToolTier:
            tier_tools = self.get_by_tier(tier)
            counts = {"available": 0, "unavailable": 0, "degraded": 0}
            for t in tier_tools:
                status = t.get_status().value
                counts[status] = counts.get(status, 0) + 1
            if tier_tools:
                summary[tier.value] = counts
        return summary

    def provider_menu(self) -> dict[str, dict[str, Any]]:
        """Capability-grouped menu of tools for the agent's preflight report.

        Returns a dict like:
        {
            "cartoonize": {
                "available": [{"name": ..., "provider": ..., "runtime": ...}],
                "unavailable": [{"name": ..., "install_instructions": ...}],
                "total": 3,
                "configured": 1,
            },
            ...
        }

        Adding a new tool to tools/ is enough; this method auto-discovers it.
        """
        self.ensure_discovered()
        menu: dict[str, dict[str, Any]] = {}

        # Skip selectors — they aggregate, they aren't providers themselves
        tools = [t for t in self._tools.values() if t.provider != "selector"]

        for tool in tools:
            cap = tool.capability
            if cap not in menu:
                menu[cap] = {"available": [], "unavailable": [], "total": 0, "configured": 0}

            status = tool.get_status()
            entry = {
                "name": tool.name,
                "provider": tool.provider,
                "runtime": tool.runtime.value,
                "best_for": tool.best_for,
                "dependencies": list(tool.dependencies),
                "required_models": list(tool.required_models),
                "install_instructions": tool.install_instructions,
                "status": status.value,
            }

            if status == ToolStatus.AVAILABLE:
                menu[cap]["available"].append(entry)
                menu[cap]["configured"] += 1
            else:
                menu[cap]["unavailable"].append(entry)
            menu[cap]["total"] += 1

        for bucket in menu.values():
            bucket["available"].sort(key=lambda entry: (entry["provider"], entry["name"]))
            bucket["unavailable"].sort(key=lambda entry: (entry["provider"], entry["name"]))

        return dict(sorted(menu.items()))

    def provider_menu_summary(self) -> dict[str, Any]:
        """Compact, human-ready rollup for onboarding/preflight.

        Example output (abbreviated):
        {
          "hardware": {"cuda": True, "gpu_name": "RTX 2060", "vram_mb": 6144, ...},
          "capabilities": [
            {"capability": "cartoonize", "configured": 2, "total": 3,
             "available_tools": ["opencv_cartoonize", "diffusers_cartoonize"],
             "unavailable_tools": ["comfyui_cartoonize"]},
            ...
          ],
          "missing_models": ["sdxl_base", ...],
          "install_hints": [{"tool": ..., "install_instructions": ...}],
        }

        Agents should present this rather than the raw support_envelope().
        """
        self.ensure_discovered()
        menu = self.provider_menu()

        capabilities: list[dict[str, Any]] = []
        install_hints: list[dict[str, Any]] = []
        for cap, bucket in menu.items():
            capabilities.append(
                {
                    "capability": cap,
                    "configured": bucket.get("configured", 0),
                    "total": bucket.get("total", 0),
                    "available_tools": [e["name"] for e in bucket.get("available", [])],
                    "unavailable_tools": [e["name"] for e in bucket.get("unavailable", [])],
                }
            )
            for entry in bucket.get("unavailable", []):
                install_hints.append(
                    {
                        "capability": cap,
                        "tool": entry["name"],
                        "install_instructions": entry.get("install_instructions") or "",
                    }
                )

        try:
            from lib.hardware import detect_hardware

            hardware = detect_hardware().as_dict()
        except Exception as exc:  # pragma: no cover - defensive
            hardware = {"error": str(exc)}

        try:
            from lib.model_registry import model_registry

            needed = sorted({m for t in self._tools.values() for m in t.required_models})
            missing_models = [m for m in needed if not model_registry.is_downloaded(m)]
        except Exception:
            missing_models = []

        result = {
            "hardware": hardware,
            "capabilities": capabilities,
            "missing_models": missing_models,
            "install_hints": install_hints,
        }
        # Normalize unicode punctuation so preflight output prints cleanly on
        # Windows cp1252 stdout. Markdown docs keep their typographic dashes.
        return _scrub_unicode_dashes(result)

    def gpu_required_tools(self) -> list[str]:
        """List tools that require GPU (VRAM > 0)."""
        return [
            t.name for t in self._tools.values()
            if t.resource_profile.vram_mb > 0
        ]

    def network_required_tools(self) -> list[str]:
        """List tools that require network access (should always be empty)."""
        return [
            t.name for t in self._tools.values()
            if t.resource_profile.network_required
        ]


# Singleton registry instance
registry = ToolRegistry()
