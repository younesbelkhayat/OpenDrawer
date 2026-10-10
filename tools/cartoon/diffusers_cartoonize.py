"""Local diffusion cartoonization with diffusers (SDXL or SD1.5).

ControlNet (Canny) img2img keeps the photo's geometry and colors; the style
prompt turns it into a children's-book illustration; IP-Adapter keeps the
face recognizable:

* ``plus_face`` — IP-Adapter Plus Face (Apache-2.0, commercial-safe)
* ``faceid``    — IP-Adapter FaceID with InsightFace embeddings (best
  resemblance, NON-commercial)

Built for small GPUs: fp16, VAE tiling, and an out-of-memory ladder
(full GPU → model offload → sequential offload) chosen from the VRAM.
All weights come from ``models.yaml`` through the model registry; nothing
is downloaded here.
"""

from __future__ import annotations

import gc
from typing import Any, Optional

import numpy as np

from lib.hardware import detect_hardware, memory_ladder, pick_memory_strategy, recommended_family
from lib.model_registry import LicenseError, model_registry
from tools.base_tool import ResourceProfile, RetryPolicy, ToolRuntime, ToolStability, ToolStatus
from tools.cartoon._common import Job, canny_control, fit_size
from tools.cartoon._provider import CartoonizeProvider

FAMILY_MODELS = {
    "sdxl": {
        "base": "sdxl_base",
        "vae": "sdxl_vae_fp16_fix",
        "controlnet": "sdxl_controlnet_canny",
        "plus_face": "ip_adapter_sdxl_plus_face",
        "faceid": "ip_adapter_faceid_sdxl",
    },
    "sd15": {
        "base": "sd15_base",
        "vae": None,
        "controlnet": "sd15_controlnet_canny",
        "plus_face": "ip_adapter_sd15_plus_face",
        "faceid": "ip_adapter_faceid_sd15",
    },
}


def _variant(path) -> Optional[str]:
    """'fp16' when the folder ships fp16 safetensors, else None."""
    return "fp16" if any(path.rglob("*.fp16.safetensors")) else None


def _is_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "out of memory" in text or "outofmemory" in text


def _insightface_available() -> bool:
    try:
        import insightface  # noqa: F401

        return True
    except Exception:
        return False


class DiffusersCartoonize(CartoonizeProvider):
    name = "diffusers_cartoonize"
    version = "1.0.0"
    provider = "diffusers"
    stability = ToolStability.EXPERIMENTAL
    runtime = ToolRuntime.LOCAL_GPU
    dependencies = ["python:torch", "python:diffusers", "python:transformers"]
    install_instructions = (
        "make install-gpu   # torch CUDA + diffusers + transformers\n"
        "python -m opendrawer models download sdxl   # ou: sd15"
    )
    agent_skills = ["diffusers-low-vram", "cartoon-prompting"]
    capabilities = ["photo_to_cartoon", "face_identity", "controlnet_canny", "low_vram"]
    best_for = ["qualité livre jeunesse", "ressemblance du visage", "GPU 6 Go et plus"]
    not_good_for = ["machine sans GPU (heures de calcul)"]
    required_models = [
        "sdxl_base",
        "sdxl_vae_fp16_fix",
        "sdxl_controlnet_canny",
        "ip_adapter_sdxl_plus_face",
        "ip_adapter_image_encoder",
    ]
    resource_profile = ResourceProfile(cpu_cores=4, ram_mb=16000, vram_mb=6000, disk_mb=12000)
    retry_policy = RetryPolicy(max_retries=0)
    fallback = "opencv_cartoonize"
    fallback_tools = ["comfyui_cartoonize", "opencv_cartoonize"]

    def get_status(self) -> ToolStatus:
        try:
            import diffusers  # noqa: F401
            import torch  # noqa: F401
        except Exception:
            return ToolStatus.UNAVAILABLE
        family = recommended_family()
        models = FAMILY_MODELS[family]
        needed = [models["base"], models["controlnet"]]
        if not all(model_registry.is_downloaded(k) for k in needed):
            return ToolStatus.UNAVAILABLE
        return ToolStatus.AVAILABLE if detect_hardware().device != "cpu" else ToolStatus.DEGRADED

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        family = inputs.get("family") or "auto"
        if family == "auto":
            family = recommended_family()
        strategy = pick_memory_strategy(family)
        per_image = {
            ("sdxl", "full_gpu"): 25,
            ("sdxl", "model_offload"): 70,
            ("sdxl", "sequential_offload"): 300,
            ("sd15", "full_gpu"): 12,
            ("sd15", "model_offload"): 25,
            ("sd15", "sequential_offload"): 90,
        }.get((family, strategy), 1800)
        return float(per_image * int(inputs.get("candidates", 3)))

    # ------------------------------------------------------------------ setup

    def _resolve_identity(self, requested: str, family: str, commercial_safe: bool, has_face: bool, warnings: list[str]) -> str:
        if not has_face or requested == "none":
            return "none"
        models = FAMILY_MODELS[family]
        if requested == "faceid":
            if commercial_safe:
                raise LicenseError("identity=faceid is non-commercial; not allowed with --commercial-safe")
            if not _insightface_available():
                raise RuntimeError("identity=faceid needs `pip install insightface onnxruntime-gpu`")
            model_registry.path(models["faceid"], commercial_safe)
            return "faceid"
        if requested == "plus_face":
            model_registry.path(models["plus_face"], commercial_safe)
            model_registry.path("ip_adapter_image_encoder", commercial_safe)
            return "plus_face"
        # auto
        if (
            not commercial_safe
            and _insightface_available()
            and model_registry.is_downloaded(models["faceid"])
        ):
            return "faceid"
        if model_registry.is_downloaded(models["plus_face"]) and model_registry.is_downloaded("ip_adapter_image_encoder"):
            return "plus_face"
        warnings.append("No IP-Adapter weights downloaded: face resemblance relies on ControlNet only")
        return "none"

    def prepare(self, inputs: dict[str, Any], analysis: dict[str, Any], jobs: list[Job]) -> dict[str, Any]:
        import torch

        hw = detect_hardware()
        family = inputs.get("family") or "auto"
        if family == "auto":
            family = recommended_family(hw)
        commercial_safe = bool(inputs.get("commercial_safe"))
        warnings: list[str] = []
        identity = self._resolve_identity(
            inputs.get("identity") or "auto",
            family,
            commercial_safe,
            any(j.face_box is not None for j in jobs),
            warnings,
        )
        ctx: dict[str, Any] = {
            "family": family,
            "identity": identity,
            "commercial_safe": commercial_safe,
            "device": hw.device,
            "dtype": torch.float16 if hw.device == "cuda" else torch.float32,
            "warnings": warnings,
            "pipe": None,
            "ladder": memory_ladder(pick_memory_strategy(family, hw)),
            "params": None,
            "face_app": None,
        }
        if identity == "faceid":
            from insightface.app import FaceAnalysis

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if hw.cuda else ["CPUExecutionProvider"]
            app = FaceAnalysis(name="buffalo_l", providers=providers)
            app.prepare(ctx_id=0 if hw.cuda else -1, det_size=(640, 640))
            ctx["face_app"] = app
        # Loading can run out of memory too (e.g. full_gpu on a card that is
        # busy): walk the ladder here as well, not only during generation.
        last_exc: Exception | None = None
        for strategy in ctx["ladder"]:
            try:
                self._build(ctx, strategy)
                return ctx
            except Exception as exc:
                if not _is_oom(exc):
                    raise
                last_exc = exc
                warnings.append(f"OOM while loading with {strategy}")
                self._free(ctx)
        raise RuntimeError(
            "Out of GPU memory while loading the models. Try --family sd15 or close other GPU applications."
        ) from last_exc

    def _build(self, ctx: dict[str, Any], strategy: str) -> None:
        import torch
        from diffusers import (
            AutoencoderKL,
            ControlNetModel,
            StableDiffusionControlNetImg2ImgPipeline,
            StableDiffusionXLControlNetImg2ImgPipeline,
        )

        self._free(ctx)
        family, dtype, cs = ctx["family"], ctx["dtype"], ctx["commercial_safe"]
        models = FAMILY_MODELS[family]
        base_dir = model_registry.path(models["base"], cs)
        cn_dir = model_registry.path(models["controlnet"], cs)
        controlnet = ControlNetModel.from_pretrained(cn_dir, torch_dtype=dtype, variant=_variant(cn_dir))
        kwargs: dict[str, Any] = {
            "controlnet": controlnet,
            "torch_dtype": dtype,
            "variant": _variant(base_dir / "unet"),
            "use_safetensors": True,
        }
        if ctx["identity"] == "plus_face":
            from transformers import CLIPVisionModelWithProjection

            enc_root = model_registry.path("ip_adapter_image_encoder", cs)
            kwargs["image_encoder"] = CLIPVisionModelWithProjection.from_pretrained(
                enc_root / "models" / "image_encoder", torch_dtype=dtype
            )
        if family == "sdxl":
            if models["vae"] and model_registry.is_downloaded(models["vae"]):
                kwargs["vae"] = AutoencoderKL.from_pretrained(model_registry.path(models["vae"], cs), torch_dtype=dtype)
            pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(base_dir, **kwargs)
        else:
            pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                base_dir, safety_checker=None, requires_safety_checker=False, **kwargs
            )

        if ctx["identity"] in ("plus_face", "faceid"):
            key = models[ctx["identity"]]
            spec = model_registry.spec(key)
            pipe.load_ip_adapter(
                str(model_registry.path(key, cs)),
                subfolder=spec.subfolder or "",
                weight_name=spec.weight_name,
                image_encoder_folder=None,
            )

        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        if strategy == "full_gpu":
            pipe.to(ctx["device"])
        elif strategy == "model_offload":
            pipe.enable_model_cpu_offload()
        elif strategy == "sequential_offload":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to("cpu")
        pipe.set_progress_bar_config(disable=True)
        ctx["pipe"] = pipe
        ctx["strategy"] = strategy

    @staticmethod
    def _free(ctx: dict[str, Any]) -> None:
        if ctx.get("pipe") is not None:
            ctx["pipe"] = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def teardown(self, ctx: dict[str, Any]) -> None:
        self._free(ctx)
        ctx["face_app"] = None

    def describe(self, ctx: dict[str, Any]) -> dict[str, Any]:
        return {
            "model_family": ctx.get("family"),
            "identity_mode": ctx.get("identity", "none"),
            "memory_strategy": ctx.get("strategy"),
            "model": FAMILY_MODELS.get(ctx.get("family", ""), {}).get("base"),
            "warnings": ctx.get("warnings", []),
        }

    # ------------------------------------------------------------------ run

    def _face_inputs(self, job: Job, crop_rgb: np.ndarray, ctx: dict[str, Any]) -> dict[str, Any]:
        import torch
        from PIL import Image

        identity = ctx["identity"]
        if identity == "none":
            return {}
        pipe = ctx["pipe"]
        scale = float(ctx["params"].get("ip_adapter_scale", 0.6))
        if job.face_box is None:
            pipe.set_ip_adapter_scale(0.0)
        else:
            pipe.set_ip_adapter_scale(scale)

        if identity == "faceid":
            emb = None
            if job.face_box is not None:
                faces = ctx["face_app"].get(crop_rgb[:, :, ::-1].copy())
                if faces:
                    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    emb = torch.from_numpy(face.normed_embedding).unsqueeze(0)
            if emb is None:
                emb = torch.zeros((1, 512))
                pipe.set_ip_adapter_scale(0.0)
            ref = emb.unsqueeze(0).unsqueeze(0)  # (1, 1, 1, 512)
            embeds = torch.cat([torch.zeros_like(ref), ref]).to(dtype=ctx["dtype"], device=ctx["device"])
            return {"ip_adapter_image_embeds": [embeds]}

        # plus_face: a square crop around the face (or the whole crop).
        h, w = crop_rgb.shape[:2]
        if job.face_box is not None:
            fx, fy, fw, fh = job.face_box
            side = int(max(fw, fh) * 1.8)
            cx, cy = fx + fw // 2, fy + fh // 2
            x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
            face = crop_rgb[y0:min(h, y0 + side), x0:min(w, x0 + side)]
        else:
            face = crop_rgb
        return {"ip_adapter_image": Image.fromarray(face).resize((224, 224))}

    def generate_job(self, job: Job, crop_rgb: np.ndarray, seed: int, ctx: dict[str, Any]) -> np.ndarray:
        import torch
        from PIL import Image

        family = ctx["family"]
        if ctx["params"] is None:
            ctx["params"] = self.style_params(ctx["inputs"], ctx["style"], family)
        params = ctx["params"]
        prompt, negative = self.prompts_for(ctx["style"], job, ctx["descriptions"])
        h, w = crop_rgb.shape[:2]
        tw, th = fit_size(w, h, int(params.get("width_long_side", 1024 if family == "sdxl" else 768)))
        init = Image.fromarray(crop_rgb).resize((tw, th), Image.LANCZOS)
        control = Image.fromarray(canny_control(np.array(init)))

        while True:
            try:
                kwargs = dict(
                    prompt=prompt,
                    negative_prompt=negative,
                    image=init,
                    control_image=control,
                    strength=float(params.get("strength", 0.7)),
                    num_inference_steps=int(params.get("steps", 30)),
                    guidance_scale=float(params.get("guidance_scale", 6.5)),
                    controlnet_conditioning_scale=float(params.get("controlnet_scale", 0.6)),
                    generator=torch.Generator("cpu").manual_seed(seed),
                )
                kwargs.update(self._face_inputs(job, crop_rgb, ctx))
                with torch.inference_mode():
                    out = ctx["pipe"](**kwargs).images[0]
                return np.array(out.convert("RGB"))
            except Exception as exc:
                if not _is_oom(exc):
                    raise
                ladder = ctx["ladder"]
                idx = ladder.index(ctx["strategy"]) if ctx.get("strategy") in ladder else len(ladder)
                if idx + 1 >= len(ladder):
                    raise RuntimeError(
                        f"Out of GPU memory even with {ctx.get('strategy')}. Try --family sd15 "
                        "or close other GPU applications."
                    ) from exc
                ctx["warnings"].append(f"OOM with {ctx['strategy']}, retrying with {ladder[idx + 1]}")
                self._build(ctx, ladder[idx + 1])
