"""Hardware detection and memory strategy selection.

Diffusion on a 6 GB card is all about choosing the right offload strategy.
This module answers two questions without importing torch at module load:

* what is on this machine (CUDA? how much VRAM?)
* given a model family, which memory strategy should a pipeline use

The strategies are ordered from fastest to most frugal; a tool that hits an
out-of-memory error walks down the ladder (see ``memory_ladder``).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Optional

# Ordered fastest → most frugal.
STRATEGIES = ("full_gpu", "model_offload", "sequential_offload", "cpu")

# Rough peak VRAM (MB, fp16) for a ControlNet + IP-Adapter img2img run with the
# whole stack resident on the GPU. Used only to pick a starting strategy.
_FULL_GPU_BUDGET_MB = {
    "sd15": 5200,
    "sdxl": 12500,
}
# Peak with model_cpu_offload (one sub-model on the GPU at a time).
_MODEL_OFFLOAD_BUDGET_MB = {
    "sd15": 3500,
    "sdxl": 7000,
}


@dataclass
class HardwareInfo:
    cuda: bool = False
    mps: bool = False
    gpu_name: Optional[str] = None
    vram_mb: int = 0
    torch_version: Optional[str] = None
    notes: list[str] = field(default_factory=list)

    @property
    def device(self) -> str:
        if self.cuda:
            return "cuda"
        if self.mps:
            return "mps"
        return "cpu"

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["device"] = self.device
        return data


@lru_cache(maxsize=1)
def detect_hardware() -> HardwareInfo:
    """Probe the machine once. Never raises."""
    info = HardwareInfo()
    forced = os.environ.get("OPENDRAWER_DEVICE", "").strip().lower()
    try:
        import torch  # noqa: F401
    except Exception:
        info.notes.append("torch not installed: GPU tools unavailable (run `make install-gpu`)")
        return info

    import torch

    info.torch_version = getattr(torch, "__version__", None)
    try:
        if torch.cuda.is_available() and forced in ("", "cuda"):
            info.cuda = True
            props = torch.cuda.get_device_properties(0)
            info.gpu_name = props.name
            info.vram_mb = int(props.total_memory // (1024 * 1024))
        elif (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
            and forced in ("", "mps")
        ):
            info.mps = True
            info.notes.append("Apple MPS: unified memory, expect slow diffusion")
    except Exception as exc:  # pragma: no cover - driver specific
        info.notes.append(f"GPU probe failed: {exc}")
    if forced == "cpu":
        info.cuda = info.mps = False
        info.notes.append("OPENDRAWER_DEVICE=cpu forces CPU execution")
    return info


def pick_memory_strategy(family: str, hw: Optional[HardwareInfo] = None) -> str:
    """Starting memory strategy for a model family on this machine."""
    hw = hw or detect_hardware()
    override = os.environ.get("OPENDRAWER_MEMORY_STRATEGY", "").strip()
    if override in STRATEGIES:
        return override
    if not (hw.cuda or hw.mps):
        return "cpu"
    if hw.mps:
        return "model_offload"
    vram = hw.vram_mb
    if vram >= _FULL_GPU_BUDGET_MB.get(family, 12500):
        return "full_gpu"
    if vram >= _MODEL_OFFLOAD_BUDGET_MB.get(family, 7000):
        return "model_offload"
    return "sequential_offload"


def memory_ladder(start: str) -> list[str]:
    """Strategies to try, from ``start`` down to the most frugal GPU option."""
    if start not in STRATEGIES:
        start = "model_offload"
    ladder = list(STRATEGIES[STRATEGIES.index(start):])
    # CPU diffusion takes hours at SDXL sizes: never fall back to it silently.
    if start != "cpu" and "cpu" in ladder:
        ladder.remove("cpu")
    return ladder


def recommended_family(hw: Optional[HardwareInfo] = None) -> str:
    """Default diffusion family: quality first, SD1.5 only without a GPU."""
    hw = hw or detect_hardware()
    if not (hw.cuda or hw.mps):
        return "sd15"
    return "sdxl"
