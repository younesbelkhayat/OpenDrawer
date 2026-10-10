"""Cartoonize contract: job planning, scoring, backends, licensing."""

from __future__ import annotations

import numpy as np
import pytest

from opendrawer.demo import draw_character
from tools.base_tool import ToolStatus
from tools.cartoon._common import compose_jobs, plan_jobs, score_candidate
from tools.cartoon.cartoonize import Cartoonize
from tools.cartoon.comfyui_cartoonize import build_workflow
from tools.cartoon.diffusers_cartoonize import DiffusersCartoonize
from tools.cartoon.opencv_cartoonize import cartoonize_opencv


def test_single_subject_is_one_job():
    _, mask = draw_character(300, 400)
    jobs = plan_jobs({"subjects": [{"id": "s1", "kind": "person", "bbox": [0, 0, 300, 400], "face_bbox": [80, 50, 140, 140]}]}, mask)
    assert len(jobs) == 1 and jobs[0].face_box == (80, 50, 140, 140)
    out = compose_jobs(mask.shape, [(jobs[0], np.zeros((400, 300, 3), np.uint8))])
    import cv2

    near = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    assert out[~near].min() == 255  # away from the (feathered) edge, outside stays white


def test_scores_prefer_faithful_candidates():
    img, mask = draw_character(450, 600)
    good = cartoonize_opencv(img, mask, seed=0)
    bad = np.full_like(img, 255)
    assert score_candidate(good, img, mask)["total"] > score_candidate(bad, img, mask)["total"]


def test_selector_falls_back_to_opencv():
    assert Cartoonize().resolve_backend("auto") in ("diffusers", "comfyui", "opencv")
    with pytest.raises(ValueError):
        Cartoonize().resolve_backend("dall-e")


def test_diffusers_refuses_faceid_in_commercial_safe_mode():
    from lib.model_registry import LicenseError

    with pytest.raises(LicenseError):
        DiffusersCartoonize()._resolve_identity("faceid", "sdxl", True, True, [])
    assert DiffusersCartoonize()._resolve_identity("auto", "sdxl", False, False, []) == "none"


def test_diffusers_unavailable_without_weights():
    assert DiffusersCartoonize().get_status() != ToolStatus.AVAILABLE


def test_comfy_workflow_graph():
    wf, out = build_workflow(
        ckpt_name="a.safetensors", controlnet_name="c.safetensors", image_name="in.png", width=768, height=1024,
        prompt="p", negative="n", seed=1, steps=20, cfg=6.0, denoise=0.7, control_strength=0.6, face_image_name="f.png",
    )
    assert wf[out]["class_type"] == "SaveImage"
    assert wf["10"]["inputs"]["model"] == ["15", 0]  # IP-Adapter feeds the sampler
    for node in wf.values():
        for value in node["inputs"].values():
            if isinstance(value, list):
                assert value[0] in wf, f"dangling link {value}"
