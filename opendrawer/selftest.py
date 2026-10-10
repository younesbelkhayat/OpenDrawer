"""Hardware/model self-test: what works on this machine, with timings.

``python -m opendrawer selftest`` checks each layer in order (Python deps,
torch/CUDA, downloaded weights, segmentation, detection, one real diffusion
generation on the synthetic demo photo) and writes a JSON report that can be
shared to debug a setup. Each check is isolated: one failure never hides the
next ones.
"""

from __future__ import annotations

import importlib
import json
import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from lib.paths import PROJECTS_DIR

Check = dict[str, Any]


def _run(name: str, fn: Callable[[], Any]) -> Check:
    started = time.monotonic()
    try:
        detail = fn()
        status = "ok"
    except Exception as exc:  # report, never raise
        status = "fail"
        detail = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=4)[-1500:]}
    return {"check": name, "status": status, "seconds": round(time.monotonic() - started, 1), "detail": detail}


def _versions() -> dict[str, Any]:
    out: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "cv2", "scipy", "skimage", "torch", "diffusers", "transformers", "accelerate",
                "rembg", "onnxruntime", "controlnet_aux", "insightface", "huggingface_hub"):
        try:
            out[mod] = getattr(importlib.import_module(mod), "__version__", "installed")
        except Exception:
            out[mod] = None
    return out


def _hardware() -> dict[str, Any]:
    from lib.hardware import detect_hardware, pick_memory_strategy, recommended_family

    hw = detect_hardware()
    info = hw.as_dict()
    info["recommended_family"] = recommended_family(hw)
    info["strategy_sdxl"] = pick_memory_strategy("sdxl", hw)
    info["strategy_sd15"] = pick_memory_strategy("sd15", hw)
    if not hw.cuda:
        raise RuntimeError(f"No CUDA GPU visible to torch ({info})")
    return info


def _models(family: str) -> dict[str, Any]:
    from lib.model_registry import model_registry

    rows = {r["key"]: r["downloaded"] for r in model_registry.status(model_registry.expand([family, "analysis"]))}
    missing = [k for k, ok in rows.items() if not ok]
    if missing:
        raise RuntimeError(f"Missing: {missing} → python -m opendrawer models download {family} analysis")
    return rows


def run_selftest(family: str = "auto", generate: bool = True, steps: int = 12) -> dict[str, Any]:
    from lib.hardware import recommended_family
    from opendrawer.demo import make_demo_photo
    from tools.tool_registry import registry

    if family == "auto":
        family = recommended_family()
    work = Path(tempfile.mkdtemp(prefix="opendrawer_selftest_"))
    photo = make_demo_photo(work / "photo.jpg")
    registry.ensure_discovered()
    checks: list[Check] = [
        _run("versions", _versions),
        _run("hardware", _hardware),
        _run(f"models:{family}", lambda: _models(family)),
    ]

    def segmentation():
        res = registry.get("bg_remove").execute(
            {"input_path": str(photo), "mask_path": str(work / "mask.png"), "cutout_path": str(work / "cutout.png")}
        )
        if not res.success:
            raise RuntimeError(res.error)
        if res.data["model"] == "grabcut":
            raise RuntimeError(f"rembg unavailable, fell back to GrabCut: {res.data['warnings']}")
        return {"model": res.data["model"], "foreground_fraction": res.data["foreground_fraction"]}

    def detection():
        res = registry.get("subject_detect").execute({"image_path": str(photo), "detector": "owlv2"})
        if not res.success:
            raise RuntimeError(res.error)
        return {"subjects": [s["label"] for s in res.data["subjects"]], "tools": res.data["tools_used"]}

    checks.append(_run("segmentation (rembg)", segmentation))
    checks.append(_run("detection (owlv2)", detection))

    if generate:
        def diffusion():
            from opendrawer.runner import AutoApprover, PageRun, RunSettings

            settings = RunSettings(backend="diffusers", family=family, candidates=1, seed=0, steps=steps)
            run = PageRun.create(photo, settings, AutoApprover(), title="selftest", projects_dir=work / "projects", log=lambda _: None)
            status = run.run()
            cs = json.loads((run.project_dir / "artifacts" / "cartoon_set.json").read_text())
            page = run.project_dir / "output" / "page.png"
            return {
                "status": status,
                "family": cs.get("model_family"),
                "identity_mode": cs.get("identity_mode"),
                "memory_strategy": cs.get("memory_strategy"),
                "cartoon": cs["candidates"][0]["path"],
                "page": str(page) if page.exists() else None,
            }

        checks.append(_run(f"diffusion ({family}, {steps} steps, full page)", diffusion))

    failed = [c["check"] for c in checks if c["status"] != "ok"]
    report = {"ok": not failed, "failed": failed, "work_dir": str(work), "checks": checks}
    out = PROJECTS_DIR / "_selftest_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(out)
    return report
