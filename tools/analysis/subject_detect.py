"""Detect the subjects of a photo: people, animals, and their faces.

Layered so it always returns something useful:

1. People/animals: OWLv2 open-vocabulary detection (transformers, Apache-2.0)
   when installed and downloaded.
2. Faces: OpenCV YuNet (tiny ONNX, MIT) when downloaded, else the Haar cascade
   bundled with OpenCV (no download at all).
3. Without OWLv2, people are inferred from faces; without any face either,
   the whole foreground becomes a single "unknown" subject.

When ``foreground_mask_path`` is given, a mask per subject is written
(foreground ∩ subject box) so later stages can process subjects one by one.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lib.imaging import load_mask, load_rgb, mask_bbox, save_mask
from lib.model_registry import ModelNotDownloaded, model_registry
from tools.base_tool import (
    BaseTool,
    Determinism,
    ResourceProfile,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)

# Text queries → (kind, label). Phrased as OWL-ViT/OWLv2 expects.
QUERIES: list[tuple[str, str, str]] = [
    ("a photo of a person", "person", "person"),
    ("a photo of a child", "person", "child"),
    ("a photo of a baby", "person", "baby"),
    ("a photo of a dog", "animal", "dog"),
    ("a photo of a cat", "animal", "cat"),
    ("a photo of a rabbit", "animal", "rabbit"),
    ("a photo of a horse", "animal", "horse"),
    ("a photo of a bird", "animal", "bird"),
]

DESCRIPTIONS = {
    "person": "a smiling person",
    "child": "a smiling child",
    "baby": "a cute baby",
    "dog": "a happy dog",
    "cat": "a cute cat",
    "rabbit": "a cute rabbit",
    "horse": "a friendly horse",
    "bird": "a cute bird",
}


def _iou(a: list[int], b: list[int]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax0 + aw, bx0 + bw), min(ay0 + ah, by0 + bh)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def nms(dets: list[dict[str, Any]], iou_threshold: float = 0.5) -> list[dict[str, Any]]:
    """Class-agnostic NMS: keep the highest-scoring box of each overlap group."""
    kept: list[dict[str, Any]] = []
    for det in sorted(dets, key=lambda d: -d["score"]):
        if all(_iou(det["bbox"], k["bbox"]) < iou_threshold for k in kept):
            kept.append(det)
    return kept


def detect_faces(rgb: np.ndarray) -> tuple[list[list[int]], str]:
    """Face boxes [x, y, w, h] and the detector name used."""
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    try:
        model_path = model_registry.path("yunet_face")
        scale = min(1.0, 1280 / max(h, w))
        small = cv2.resize(bgr, (int(w * scale), int(h * scale))) if scale < 1 else bgr
        detector = cv2.FaceDetectorYN.create(str(model_path), "", (small.shape[1], small.shape[0]), 0.7)
        _, faces = detector.detect(small)
        boxes = []
        if faces is not None:
            for f in faces:
                x, y, fw, fh = (f[:4] / scale).astype(int).tolist()
                boxes.append([max(0, x), max(0, y), fw, fh])
        return boxes, "yunet"
    except (ModelNotDownloaded, AttributeError, cv2.error, KeyError):
        pass
    if not hasattr(cv2, "CascadeClassifier"):  # removed from OpenCV 5 main modules
        return [], "none"
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    min_side = max(24, int(min(h, w) * 0.05))
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(min_side, min_side))
    return [list(map(int, f)) for f in faces], "haar"


def detect_owlv2(rgb: np.ndarray, threshold: float = 0.25) -> list[dict[str, Any]]:
    """Open-vocabulary detection with OWLv2. Raises if unavailable."""
    import torch
    from PIL import Image
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    path = str(model_registry.path("owlv2"))
    processor = Owlv2Processor.from_pretrained(path)
    model = Owlv2ForObjectDetection.from_pretrained(path).eval()
    texts = [q for q, _, _ in QUERIES]
    image = Image.fromarray(rgb)
    inputs = processor(text=[texts], images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    side = max(rgb.shape[:2])  # OWLv2 pads to a square before resizing
    target = torch.tensor([[side, side]])
    post = getattr(processor, "post_process_grounded_object_detection", None) or processor.post_process_object_detection
    result = post(outputs=outputs, threshold=threshold, target_sizes=target)[0]
    h, w = rgb.shape[:2]
    dets = []
    for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
        x0, y0, x1, y1 = [float(v) for v in box.tolist()]
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(w), x1), min(float(h), y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        _, kind, name = QUERIES[int(label)]
        dets.append(
            {"kind": kind, "label": name, "score": float(score), "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]}
        )
    return nms(dets)


def people_from_faces(faces: list[list[int]], width: int, height: int) -> list[dict[str, Any]]:
    """Rough body boxes from face boxes (head-and-shoulders down to the frame)."""
    out = []
    for x, y, fw, fh in faces:
        x0 = max(0, int(x - 1.6 * fw))
        x1 = min(width, int(x + fw + 1.6 * fw))
        y0 = max(0, int(y - 0.8 * fh))
        out.append({"kind": "person", "label": "person", "score": 0.5, "bbox": [x0, y0, x1 - x0, height - y0]})
    return out


def assign_faces(subjects: list[dict[str, Any]], faces: list[list[int]]) -> None:
    for subject in subjects:
        subject["face_bbox"] = None
    for face in faces:
        cx, cy = face[0] + face[2] / 2, face[1] + face[3] / 2
        best, best_area = None, None
        for subject in subjects:
            if subject["kind"] != "person" or subject["face_bbox"] is not None:
                continue
            x, y, w, h = subject["bbox"]
            if x <= cx <= x + w and y <= cy <= y + h and (best_area is None or w * h < best_area):
                best, best_area = subject, w * h
        if best is not None:
            best["face_bbox"] = face


class SubjectDetect(BaseTool):
    name = "subject_detect"
    version = "1.0.0"
    tier = ToolTier.ANALYZE
    capability = "subject_detection"
    provider = "opendrawer"
    stability = ToolStability.BETA
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.HYBRID

    dependencies = ["python:cv2"]
    install_instructions = (
        "Fonctionne avec OpenCV seul (visages Haar). Pour détecter animaux et personnes :\n"
        "pip install transformers torch && python -m opendrawer models download owlv2 yunet_face"
    )
    capabilities = ["person_detection", "animal_detection", "face_detection", "subject_masks"]
    best_for = ["photos de famille, enfants, animaux de compagnie"]
    required_models = ["owlv2", "yunet_face"]

    input_schema = {
        "type": "object",
        "required": ["image_path"],
        "properties": {
            "image_path": {"type": "string"},
            "foreground_mask_path": {"type": "string", "description": "Optional, enables per-subject masks"},
            "mask_dir": {"type": "string", "description": "Where per-subject masks are written"},
            "detector": {"type": "string", "enum": ["auto", "owlv2", "faces"], "default": "auto"},
            "threshold": {"type": "number", "default": 0.25},
            "max_subjects": {"type": "integer", "default": 6},
        },
    }
    resource_profile = ResourceProfile(cpu_cores=2, ram_mb=2500, vram_mb=0)
    idempotency_key_fields = ["image_path", "detector", "threshold"]

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        started = time.monotonic()
        path = Path(inputs["image_path"])
        if not path.exists():
            return ToolResult(success=False, error=f"Input not found: {path}")
        rgb = load_rgb(path)
        h, w = rgb.shape[:2]
        detector = inputs.get("detector", "auto")
        warnings: list[str] = []
        tools_used: list[str] = []

        subjects: list[dict[str, Any]] = []
        if detector in ("auto", "owlv2"):
            try:
                subjects = detect_owlv2(rgb, float(inputs.get("threshold", 0.25)))
                tools_used.append("owlv2")
            except Exception as exc:
                if detector == "owlv2":
                    return ToolResult(success=False, error=f"OWLv2 unavailable: {exc}")
                warnings.append(f"OWLv2 unavailable ({type(exc).__name__}); people inferred from faces")

        faces, face_method = detect_faces(rgb)
        tools_used.append(face_method)
        if not subjects:
            subjects = people_from_faces(faces, w, h)

        fg = None
        if inputs.get("foreground_mask_path"):
            fg = load_mask(inputs["foreground_mask_path"])
            if fg.shape != (h, w):
                fg = None
                warnings.append("foreground mask size mismatch; ignored")
        if not subjects:
            bbox = mask_bbox(fg) if fg is not None else None
            subjects = [
                {"kind": "unknown", "label": "subject", "score": 0.0, "bbox": list(bbox or (0, 0, w, h))}
            ]
            warnings.append("No person/animal detected; using the whole foreground as one subject")

        subjects = sorted(subjects, key=lambda s: -s["score"])[: int(inputs.get("max_subjects", 6))]
        subjects.sort(key=lambda s: s["bbox"][0])  # left to right
        assign_faces(subjects, faces)

        for i, subject in enumerate(subjects, start=1):
            subject["id"] = f"s{i}"
            subject["description"] = DESCRIPTIONS.get(subject["label"], "a smiling person")
            if fg is not None and inputs.get("mask_dir"):
                x, y, bw, bh = subject["bbox"]
                pad_x, pad_y = int(bw * 0.08), int(bh * 0.08)
                box = np.zeros_like(fg)
                box[max(0, y - pad_y):min(h, y + bh + pad_y), max(0, x - pad_x):min(w, x + bw + pad_x)] = True
                sub_mask = fg & box
                mask_path = Path(inputs["mask_dir"]) / f"subject_{subject['id']}.png"
                save_mask(sub_mask, mask_path)
                subject["mask_path"] = str(mask_path)

        return ToolResult(
            success=True,
            data={
                "image_size": {"width": w, "height": h},
                "subjects": subjects,
                "faces": faces,
                "tools_used": tools_used,
                "warnings": warnings,
            },
            duration_seconds=time.monotonic() - started,
        )
