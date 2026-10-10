"""Cross-file contracts: manifests ↔ tools ↔ skills ↔ schemas ↔ models."""

from __future__ import annotations

from pathlib import Path

import pytest

from lib.model_registry import ModelRegistry
from lib.pipeline_loader import get_required_tools, list_pipelines, load_pipeline
from schemas.artifacts import ARTIFACT_NAMES, load_schema
from tools.tool_registry import registry

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def tools():
    registry.ensure_discovered()
    return {name: registry.get(name) for name in registry.list_all()}


def test_tool_contract_fields(tools):
    assert len(tools) >= 12
    for name, tool in tools.items():
        info = tool.get_info()
        assert info["name"] == name
        assert info["capability"] != "generic", name
        assert not tool.resource_profile.network_required, f"{name} must run offline"
        assert isinstance(tool.input_schema, dict) and tool.input_schema.get("type") == "object", name


def test_manifest_tools_exist(tools):
    for pipeline in list_pipelines():
        missing = get_required_tools(load_pipeline(pipeline)) - set(tools)
        assert not missing, f"{pipeline}: unknown tools {missing}"


def test_manifest_skills_and_artifacts_exist():
    for pipeline in list_pipelines():
        manifest = load_pipeline(pipeline)
        for skill in manifest.get("required_skills", []):
            assert (ROOT / "skills" / f"{skill}.md").is_file(), skill
        for stage in manifest["stages"]:
            assert (ROOT / "skills" / f"{stage['skill']}.md").is_file(), stage["skill"]
            for artifact in stage.get("produces", []):
                assert artifact in ARTIFACT_NAMES
                load_schema(artifact)
        for mode in manifest.get("production_modes", []):
            for skill in mode.get("agent_skills", []):
                assert (ROOT / ".agents" / "skills" / skill / "SKILL.md").is_file(), skill


def test_tool_agent_skills_and_models_exist(tools):
    models = ModelRegistry()
    for name, tool in tools.items():
        for skill in tool.agent_skills:
            assert (ROOT / ".agents" / "skills" / skill / "SKILL.md").is_file(), f"{name} → {skill}"
        for key in tool.required_models:
            models.spec(key)


def test_no_video_leftovers():
    banned = ("remotion", "hyperframes", "openmontage")
    for path in list((ROOT / "tools").rglob("*.py")) + list((ROOT / "lib").rglob("*.py")) + list((ROOT / "skills").rglob("*.md")):
        text = path.read_text(encoding="utf-8").lower()
        for word in banned:
            assert word not in text, f"{path} mentions {word}"
