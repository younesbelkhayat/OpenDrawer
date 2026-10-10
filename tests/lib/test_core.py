"""Checkpoints, model registry, difficulty, hardware, styles."""

from __future__ import annotations

import pytest

from lib.checkpoint import (
    CheckpointValidationError,
    get_next_stage,
    init_project,
    write_checkpoint,
)
from lib.difficulty import PRESETS, get_preset
from lib.hardware import HardwareInfo, memory_ladder, pick_memory_strategy
from lib.model_registry import LicenseError, ModelNotDownloaded, ModelRegistry
from lib.pipeline_loader import list_pipelines, load_pipeline
from styles.style_loader import build_prompt, list_styles, load_style

PHOTO_ANALYSIS = {
    "version": "1.0",
    "source_photo": "x.jpg",
    "image_size": {"width": 10, "height": 10},
    "subjects": [{"id": "s1", "kind": "person", "bbox": [0, 0, 5, 5]}],
    "cutout_path": "a.png",
    "foreground_mask_path": "b.png",
    "palette": [{"hex": "#AABBCC"}],
}


def test_pipelines_validate():
    assert {"coloring-page", "coloring-book"} <= set(list_pipelines())
    for name in list_pipelines():
        load_pipeline(name)


def test_checkpoint_flow_and_gates(tmp_path):
    init_project("p", title="P", pipeline_type="coloring-page", pipeline_dir=tmp_path, style="livre-jeunesse")
    assert get_next_stage(tmp_path, "p") == "intake"
    with pytest.raises(CheckpointValidationError):
        write_checkpoint(tmp_path, "p", "intake", "completed", {})  # missing produced artifact
    write_checkpoint(tmp_path, "p", "intake", "completed", {"photo_analysis": PHOTO_ANALYSIS})
    assert get_next_stage(tmp_path, "p") == "cartoon"
    cartoon = {"cartoon_set": {"version": "1.0", "backend": "opencv", "candidates": [{"id": "c1", "path": "x"}]}}
    with pytest.raises(CheckpointValidationError, match="GATE VIOLATION"):
        write_checkpoint(tmp_path, "p", "cartoon", "completed", cartoon)
    write_checkpoint(tmp_path, "p", "cartoon", "completed", cartoon, human_approved=True)
    with pytest.raises(CheckpointValidationError, match="PREREQUISITE"):
        write_checkpoint(tmp_path, "p", "regions", "completed", {"region_map": {}})


def test_unknown_style_rejected(tmp_path):
    with pytest.raises(CheckpointValidationError):
        init_project("p", title="P", pipeline_type="coloring-page", pipeline_dir=tmp_path, style="nope")


def test_model_registry_licensing(tmp_path):
    reg = ModelRegistry(models_dir=tmp_path)
    assert "sdxl_base" in reg.expand(["sdxl"])
    reg.check_allowed("sdxl_base", commercial_safe=True)
    with pytest.raises(LicenseError):
        reg.check_allowed("ip_adapter_faceid_sdxl", commercial_safe=True)
    with pytest.raises(LicenseError):
        reg.check_allowed("lineart_annotators", commercial_safe=True)  # unverified licence
    with pytest.raises(ModelNotDownloaded, match="models download"):
        reg.path("sdxl_base", commercial_safe=False)


def test_every_preset_entry_exists():
    reg = ModelRegistry()
    for preset, keys in reg.presets().items():
        for key in keys:
            reg.spec(key)


def test_difficulty_aliases_and_monotony():
    assert get_preset("facile").name == "easy"
    assert get_preset("DIFFICILE").name == "hard"
    assert get_preset(None).name == "medium"
    with pytest.raises(ValueError):
        get_preset("expert")
    e, m, h = PRESETS["easy"], PRESETS["medium"], PRESETS["hard"]
    assert e.line_px > m.line_px > h.line_px
    assert e.max_regions < m.max_regions < h.max_regions


def test_memory_strategy_for_6gb():
    six = HardwareInfo(cuda=True, vram_mb=6144)
    assert pick_memory_strategy("sdxl", six) == "sequential_offload"
    assert pick_memory_strategy("sd15", six) == "full_gpu"
    assert pick_memory_strategy("sdxl", HardwareInfo()) == "cpu"
    assert memory_ladder("full_gpu") == ["full_gpu", "model_offload", "sequential_offload"]


def test_style_prompt():
    assert "livre-jeunesse" in list_styles()
    pos, neg = build_prompt(load_style("livre-jeunesse"), "dog", None)
    assert "a happy dog" in pos and "photo" in neg
