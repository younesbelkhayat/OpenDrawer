"""Cartoonization through a local ComfyUI server.

Same recipe as ``diffusers_cartoonize`` (ControlNet Canny img2img + style
prompt), expressed as a ComfyUI API workflow built in Python from core nodes
only. When the ComfyUI_IPAdapter_plus custom nodes are installed, face
conditioning is added automatically.

Model file names refer to the user's ComfyUI ``models/`` folders; set them
with ``COMFYUI_CARTOON_CHECKPOINT`` / ``COMFYUI_CARTOON_CONTROLNET`` or the
``ckpt_name`` / ``controlnet_name`` inputs.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

from lib.imaging import load_rgb, save_rgb
from tools._comfyui.client import ComfyUIClient
from tools.base_tool import ResourceProfile, ToolRuntime, ToolStability, ToolStatus
from tools.cartoon._common import Job, fit_size
from tools.cartoon._provider import CARTOONIZE_INPUT_SCHEMA, CartoonizeProvider

DEFAULT_MODELS = {
    "sdxl": ("sd_xl_base_1.0.safetensors", "controlnet-canny-sdxl-1.0-small.safetensors"),
    "sd15": ("v1-5-pruned-emaonly.safetensors", "control_v11p_sd15_canny.safetensors"),
}


def build_workflow(
    *,
    ckpt_name: str,
    controlnet_name: str,
    image_name: str,
    width: int,
    height: int,
    prompt: str,
    negative: str,
    seed: int,
    steps: int,
    cfg: float,
    denoise: float,
    control_strength: float,
    face_image_name: Optional[str] = None,
    ip_weight: float = 0.6,
) -> tuple[dict[str, Any], str]:
    """ComfyUI API-format workflow and the id of its SaveImage node."""
    wf: dict[str, Any] = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt_name}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "5": {
            "class_type": "ImageScale",
            "inputs": {"image": ["4", 0], "upscale_method": "lanczos", "width": width, "height": height, "crop": "disabled"},
        },
        "6": {"class_type": "Canny", "inputs": {"image": ["5", 0], "low_threshold": 0.3, "high_threshold": 0.7}},
        "7": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": controlnet_name}},
        "8": {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": ["2", 0],
                "negative": ["3", 0],
                "control_net": ["7", 0],
                "image": ["6", 0],
                "strength": control_strength,
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
        },
        "9": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["1", 2]}},
        "10": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "positive": ["8", 0],
                "negative": ["8", 1],
                "latent_image": ["9", 0],
                "denoise": denoise,
            },
        },
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "12": {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": "opendrawer_cartoon"}},
    }
    if face_image_name:
        wf["13"] = {
            "class_type": "IPAdapterUnifiedLoader",
            "inputs": {"model": ["1", 0], "preset": "PLUS FACE (portraits)"},
        }
        wf["14"] = {"class_type": "LoadImage", "inputs": {"image": face_image_name}}
        wf["15"] = {
            "class_type": "IPAdapter",
            "inputs": {
                "model": ["13", 0],
                "ipadapter": ["13", 1],
                "image": ["14", 0],
                "weight": ip_weight,
                "start_at": 0.0,
                "end_at": 1.0,
                "weight_type": "standard",
            },
        }
        wf["10"]["inputs"]["model"] = ["15", 0]
    return wf, "12"


class ComfyUICartoonize(CartoonizeProvider):
    name = "comfyui_cartoonize"
    version = "1.0.0"
    provider = "comfyui"
    stability = ToolStability.EXPERIMENTAL
    runtime = ToolRuntime.LOCAL_GPU
    dependencies = ["python:requests"]
    install_instructions = (
        "Lancez ComfyUI (https://github.com/comfyanonymous/ComfyUI) puis définissez\n"
        "COMFYUI_SERVER_URL (défaut http://localhost:8188), COMFYUI_CARTOON_CHECKPOINT et\n"
        "COMFYUI_CARTOON_CONTROLNET. Optionnel : ComfyUI_IPAdapter_plus pour la ressemblance."
    )
    agent_skills = ["comfyui", "cartoon-prompting"]
    capabilities = ["photo_to_cartoon", "controlnet_canny", "custom_workflow"]
    best_for = ["utilisateurs de ComfyUI", "modèles/LoRA cartoon personnels"]
    resource_profile = ResourceProfile(cpu_cores=2, ram_mb=2000, vram_mb=6000)
    fallback = "opencv_cartoonize"
    fallback_tools = ["diffusers_cartoonize", "opencv_cartoonize"]

    input_schema = {
        **CARTOONIZE_INPUT_SCHEMA,
        "properties": {
            **CARTOONIZE_INPUT_SCHEMA["properties"],
            "ckpt_name": {"type": "string", "description": "Checkpoint file name in ComfyUI"},
            "controlnet_name": {"type": "string", "description": "Canny ControlNet file name in ComfyUI"},
            "timeout_seconds": {"type": "integer", "default": 1800},
        },
    }

    def __init__(self) -> None:
        self._client = ComfyUIClient(capability="image")

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE if self._client.is_available() else ToolStatus.UNAVAILABLE

    def prepare(self, inputs: dict[str, Any], analysis: dict[str, Any], jobs: list[Job]) -> dict[str, Any]:
        if not self._client.is_available():
            raise RuntimeError(self._client.unavailable_reason())
        if inputs.get("identity") == "faceid":
            raise RuntimeError("identity=faceid is only supported by diffusers_cartoonize")
        family = inputs.get("family") or "auto"
        if family == "auto":
            family = "sdxl"
        default_ckpt, default_cn = DEFAULT_MODELS[family]
        ckpt = inputs.get("ckpt_name") or os.environ.get("COMFYUI_CARTOON_CHECKPOINT") or default_ckpt
        cn = inputs.get("controlnet_name") or os.environ.get("COMFYUI_CARTOON_CONTROLNET") or default_cn
        warnings: list[str] = []
        use_ip = inputs.get("identity") != "none" and self._client.has_node("IPAdapterUnifiedLoader")
        if inputs.get("identity") == "plus_face" and not use_ip:
            raise RuntimeError("identity=plus_face needs the ComfyUI_IPAdapter_plus custom nodes")
        if not use_ip and inputs.get("identity") != "none":
            warnings.append("ComfyUI_IPAdapter_plus not installed: no face conditioning")
        return {
            "family": family,
            "ckpt": ckpt,
            "controlnet": cn,
            "use_ip": use_ip,
            "warnings": warnings,
            "tmp": Path(tempfile.mkdtemp(prefix="opendrawer_comfy_")),
            "timeout": int(inputs.get("timeout_seconds", 1800)),
        }

    def describe(self, ctx: dict[str, Any]) -> dict[str, Any]:
        return {
            "model_family": ctx.get("family"),
            "identity_mode": "plus_face" if ctx.get("use_ip") else "none",
            "model": ctx.get("ckpt"),
            "warnings": ctx.get("warnings", []),
        }

    def generate_job(self, job: Job, crop_rgb: np.ndarray, seed: int, ctx: dict[str, Any]) -> np.ndarray:
        params = self.style_params(ctx["inputs"], ctx["style"], ctx["family"])
        prompt, negative = self.prompts_for(ctx["style"], job, ctx["descriptions"])
        h, w = crop_rgb.shape[:2]
        tw, th = fit_size(w, h, int(params.get("width_long_side", 1024)))
        tmp: Path = ctx["tmp"]
        src = save_rgb(crop_rgb, tmp / f"crop_{seed}.png")
        image_name = self._client.upload_image(src, src.name)
        face_name = None
        if ctx["use_ip"] and job.face_box is not None:
            fx, fy, fw, fh = job.face_box
            side = int(max(fw, fh) * 1.8)
            x0, y0 = max(0, fx + fw // 2 - side // 2), max(0, fy + fh // 2 - side // 2)
            face = save_rgb(crop_rgb[y0:y0 + side, x0:x0 + side], tmp / f"face_{seed}.png")
            face_name = self._client.upload_image(face, face.name)
        workflow, output_node = build_workflow(
            ckpt_name=ctx["ckpt"],
            controlnet_name=ctx["controlnet"],
            image_name=image_name,
            width=tw,
            height=th,
            prompt=prompt,
            negative=negative,
            seed=seed,
            steps=int(params.get("steps", 30)),
            cfg=float(params.get("guidance_scale", 6.5)),
            denoise=float(params.get("strength", 0.7)),
            control_strength=float(params.get("controlnet_scale", 0.6)),
            face_image_name=face_name,
            ip_weight=float(params.get("ip_adapter_scale", 0.6)),
        )
        paths = self._client.generate(workflow, output_node, tmp / f"out_{seed}.png", timeout=ctx["timeout"])
        return load_rgb(paths[0])
