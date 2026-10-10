"""Turn raw lines into closed, child-friendly zones with expected colors.

Algorithm (all at the work resolution):

1. Seal small gaps in the raw lines, then label the free (non-line) pixels of
   the subject: each connected component is a candidate zone.
2. Give every line pixel to its nearest zone, so zones tile the subject with
   no gaps (a partition). Adjacency and shared border lengths come for free.
3. Simplify with a union-find:
   * tiny *dark* zones (pupils, nostrils) become black fills, like in real
     coloring books;
   * other zones under the minimum area merge into the neighbor they share
     the longest border with;
   * while there are more zones than the difficulty allows, the most similar
     adjacent pair (in LAB, small zones first) is merged.
4. Redraw clean lines as the borders of the final partition with a uniform,
   difficulty-dependent stroke (thicker outline), plus optional open detail
   strokes (smile, folds) kept from the raw lines.
5. Color each zone from the cartoon, snapped to the photo's real palette.
"""

from __future__ import annotations

import heapq
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy import ndimage

from lib.difficulty import DifficultyPreset, get_preset
from lib.imaging import (
    delta_e,
    disk,
    hex_to_rgb,
    lab_to_hex,
    load_mask,
    load_rgb,
    nearest_crayon,
    resize_to,
    rgb_to_lab,
    save_mask,
    save_rgb,
)
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

BLACK_HEX = "#1A1A1A"
DARK_L = 32.0  # L* below which a small zone counts as "dark"
PHOTO_SNAP_MAX_DE = 22.0
MIN_BLACK_CIRCULARITY = 0.45  # pupils/nostrils are compact; slivers at line junctions are not
MIN_BLACK_ENCLOSURE = 0.9  # ...and islands inside ONE zone; ink blobs at line junctions touch several
# A large compact dark zone may touch a second zone (pupil against an eyelid).
BIG_DARK_FRAC = 0.001
BIG_DARK_MIN_ENCLOSURE = 0.6
MAX_DETAIL_HALF_WIDTH_PX = 14  # wider dark bands are areas (hair, fur), not strokes


# ------------------------------------------------------------------ helpers

def encode_labels(labels: np.ndarray) -> np.ndarray:
    """int label map → RGB (24-bit id) for lossless PNG storage."""
    lab = labels.astype(np.uint32)
    return np.stack([lab & 255, (lab >> 8) & 255, (lab >> 16) & 255], axis=2).astype(np.uint8)


def decode_labels(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.int64)
    return (rgb[..., 0] | (rgb[..., 1] << 8) | (rgb[..., 2] << 16)).astype(np.int32)


def border_mask(labels: np.ndarray) -> np.ndarray:
    """Pixels on a border between two different labels (both sides)."""
    b = np.zeros(labels.shape, bool)
    dx = labels[:, :-1] != labels[:, 1:]
    dy = labels[:-1, :] != labels[1:, :]
    b[:, :-1] |= dx
    b[:, 1:] |= dx
    b[:-1, :] |= dy
    b[1:, :] |= dy
    return b


def adjacency(partition: np.ndarray, n: int) -> dict[int, dict[int, int]]:
    """Shared border length between every pair of touching labels (0 = background)."""
    pairs = []
    for a, b in (
        (partition[:, :-1], partition[:, 1:]),
        (partition[:-1, :], partition[1:, :]),
    ):
        diff = a != b
        lo = np.minimum(a[diff], b[diff]).astype(np.int64)
        hi = np.maximum(a[diff], b[diff]).astype(np.int64)
        pairs.append(lo * (n + 1) + hi)
    codes, counts = np.unique(np.concatenate(pairs), return_counts=True) if pairs else ([], [])
    adj: dict[int, dict[int, int]] = {i: {} for i in range(n + 1)}
    for code, count in zip(codes, counts):
        lo, hi = divmod(int(code), n + 1)
        adj[lo][hi] = adj[lo].get(hi, 0) + int(count)
        adj[hi][lo] = adj[hi].get(lo, 0) + int(count)
    return adj


class ZoneGraph:
    """Union-find over zones with area, mean LAB color and border lengths."""

    def __init__(
        self,
        n: int,
        area: np.ndarray,
        color_sum: np.ndarray,
        color_w: np.ndarray,
        adj: dict[int, dict[int, int]],
        thickness: np.ndarray | None = None,
    ):
        self.parent = list(range(n + 1))
        self.area = area.astype(np.float64)
        # Radius of the largest disk that fits in the zone (crayon-ability).
        self.thickness = (
            thickness.astype(np.float64) if thickness is not None else np.full(n + 1, np.inf)
        )
        self.color_sum = color_sum.astype(np.float64)
        self.color_w = color_w.astype(np.float64)
        self.adj = adj
        self.black: set[int] = set()
        self.version = [0] * (n + 1)

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def color(self, r: int) -> np.ndarray:
        w = self.color_w[r]
        return self.color_sum[r] / w if w > 0 else np.array([100.0, 0.0, 0.0])

    def active(self) -> list[int]:
        return [r for r in range(1, len(self.parent)) if self.parent[r] == r]

    def merge(self, a: int, into: int) -> int:
        a, into = self.find(a), self.find(into)
        if a == into:
            return into
        self.parent[a] = into
        self.area[into] += self.area[a]
        self.color_sum[into] += self.color_sum[a]
        self.color_w[into] += self.color_w[a]
        self.thickness[into] = max(self.thickness[into], self.thickness[a])
        for c, length in self.adj.pop(a, {}).items():
            if c == into:
                continue
            self.adj[into][c] = self.adj[into].get(c, 0) + length
            nb = self.adj.get(c)
            if nb is not None:
                nb.pop(a, None)
                nb[into] = nb.get(into, 0) + length
        self.adj[into].pop(a, None)
        self.version[into] += 1
        self.version[a] += 1
        return into

    def zone_count(self) -> int:
        return sum(1 for r in self.active() if r not in self.black)

    def enclosure(self, r: int) -> float:
        """Share of the zone's border shared with its main neighbor (1.0 = island)."""
        borders = self.adj.get(r, {})
        total = float(sum(borders.values()))
        return max(borders.values()) / total if total else 0.0

    def circularity(self, r: int) -> float:
        """4πA/P² with P = shared border length (4-neighborhood): ~0.6 disk, ~0.8 square."""
        perimeter = float(sum(self.adj.get(r, {}).values()))
        return 4.0 * np.pi * self.area[r] / (perimeter * perimeter) if perimeter else 0.0


def simplify(graph: ZoneGraph, preset: DifficultyPreset, subject_area: float) -> None:
    min_area = preset.min_region_frac * subject_area
    dark_max = preset.dark_fill_max_frac * subject_area
    min_thickness = preset.line_px * 0.9  # thinner zones cannot be colored: slivers

    def is_sliver(r: int) -> bool:
        return graph.thickness[r] < min_thickness

    # Pass 1: small compact dark zones → black fill; other small zones and
    # slivers merge into the neighbor they share the longest border with.
    # Slivers sort first (key 0) whatever their area.
    heap = [(0.0 if is_sliver(r) else graph.area[r], r) for r in graph.active()]
    heapq.heapify(heap)
    while heap:
        key, r = heapq.heappop(heap)
        if graph.parent[r] != r or r in graph.black:
            continue
        sliver = is_sliver(r)
        if key != (0.0 if sliver else graph.area[r]):
            continue
        area = graph.area[r]
        if not sliver and area >= max(min_area, dark_max):
            break
        l_star = graph.color(r)[0]
        enclosure = graph.enclosure(r)
        if (
            not sliver
            and l_star < DARK_L
            and area <= dark_max
            and graph.circularity(r) >= MIN_BLACK_CIRCULARITY
            and (
                enclosure >= MIN_BLACK_ENCLOSURE
                or (enclosure >= BIG_DARK_MIN_ENCLOSURE and area >= BIG_DARK_FRAC * subject_area)
            )
        ):
            graph.black.add(r)
            continue
        if not sliver and area >= min_area:
            continue  # big enough to color; only black-fill eligibility mattered
        neighbors = {c: length for c, length in graph.adj.get(r, {}).items() if c not in graph.black}
        inner = {c: length for c, length in neighbors.items() if c != 0}
        if inner:
            target = max(inner, key=inner.get)
        elif 0 in neighbors:
            target = 0
        else:
            # Surrounded only by black fills: blacken it too if dark, else keep.
            if l_star < DARK_L:
                graph.black.add(r)
            continue
        if target == 0:
            # A speck on the silhouette with no other neighbor: drop it.
            graph.parent[r] = 0
            graph.adj.pop(r, None)
            for nb in graph.adj.values():
                nb.pop(r, None)
            continue
        root = graph.merge(r, target)
        heapq.heappush(heap, (0.0 if is_sliver(root) else graph.area[root], root))

    # Pass 2: too many zones → merge the most similar neighbors (small first).
    def pair_cost(a: int, b: int) -> float:
        de = float(delta_e(graph.color(a), graph.color(b)))
        small = min(graph.area[a], graph.area[b]) / max(min_area, 1.0)
        return de * (small ** 0.35)

    count = graph.zone_count()
    if count <= preset.max_regions:
        return
    heap2: list[tuple[float, int, int, int, int]] = []
    for a in graph.active():
        if a in graph.black:
            continue
        for b in graph.adj.get(a, {}):
            if b > a and b not in graph.black and b != 0:
                heap2.append((pair_cost(a, b), a, b, graph.version[a], graph.version[b]))
    heapq.heapify(heap2)
    while heap2 and count > preset.max_regions:
        _, a, b, va, vb = heapq.heappop(heap2)
        if graph.parent[a] != a or graph.parent[b] != b or graph.version[a] != va or graph.version[b] != vb:
            continue
        keep, drop = (a, b) if graph.area[a] >= graph.area[b] else (b, a)
        root = graph.merge(drop, keep)
        count -= 1
        for c in graph.adj.get(root, {}):
            if c != 0 and c not in graph.black and graph.parent[c] == c:
                lo, hi = min(root, c), max(root, c)
                heapq.heappush(heap2, (pair_cost(lo, hi), lo, hi, graph.version[lo], graph.version[hi]))


def antialiased(mask: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """Float coverage in [0, 1] with soft edges, for compositing strokes."""
    return np.clip(cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma), 0.0, 1.0)


def detail_strokes(strokes: np.ndarray, final_borders: np.ndarray, subject: np.ndarray, preset: DifficultyPreset) -> np.ndarray:
    """Open drawn strokes (smile, folds) that no zone border already draws."""
    from skimage.morphology import skeletonize

    drawn = strokes & subject
    skel = skeletonize(drawn)
    # A stroke is already rendered when its *body* (not just its centerline)
    # reaches a zone border: compare the distance to borders with the local
    # stroke half-width.
    half_width = ndimage.distance_transform_edt(drawn)
    to_border = ndimage.distance_transform_edt(~final_borders)
    skel &= to_border > half_width + preset.line_px
    n, labels = cv2.connectedComponents(skel.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(skel)
    index = np.arange(1, n)
    length = ndimage.sum(np.ones_like(half_width), labels=labels, index=index)
    width = ndimage.median(half_width, labels=labels, index=index)
    distance = ndimage.mean(to_border, labels=labels, index=index)
    keep = np.zeros(n, bool)
    # A detail is a long, *thin* stroke (wide dark bands are areas, not lines)
    # lying well inside a zone (leftovers hugging a border are outline debris).
    keep[1:] = (
        (length >= preset.min_detail_len_px)
        & (width <= MAX_DETAIL_HALF_WIDTH_PX)
        & (distance >= 3 * preset.line_px)
    )
    return keep[labels]


def ink_dots(strokes: np.ndarray, final_borders: np.ndarray, subject: np.ndarray, max_area: float) -> np.ndarray:
    """Small compact drawn blobs (nostrils, freckles) kept as solid black dots."""
    blobs = strokes & subject
    n, labels, stats, _ = cv2.connectedComponentsWithStats(blobs.astype(np.uint8), connectivity=8)
    near_border = ndimage.distance_transform_edt(~final_borders)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 12 or area > max_area:
            continue
        comp = (labels[y:y + bh, x:x + bw] == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = sum(cv2.arcLength(c, True) for c in contours) or 1.0
        if 4.0 * np.pi * area / (perimeter * perimeter) < 0.55:
            continue
        if near_border[y:y + bh, x:x + bw][comp > 0].min() < 2:
            continue  # touches a zone border: already part of a line
        keep[i] = True
    return keep[labels]


def _compact(labels: np.ndarray, origin: dict[int, int]) -> tuple[np.ndarray, dict[int, int]]:
    """Relabel to 1..K (0 stays background); carry the original zone ids along."""
    uniq = np.unique(labels)
    uniq = uniq[uniq != 0]
    remap = np.zeros(int(labels.max()) + 1, np.int32)
    remap[uniq] = np.arange(1, len(uniq) + 1, dtype=np.int32)
    return remap[labels], {int(remap[u]): origin[int(u)] for u in uniq}


def smooth_labels(labels: np.ndarray, preset: DifficultyPreset) -> np.ndarray:
    """Majority-style smoothing of zone borders (removes pixel-level wobble).

    A median filter on the label map always returns a label present in the
    window, so on a border between two zones it acts as a majority vote.
    Requires ≤ 255 labels (true after simplification); otherwise a no-op.
    """
    if labels.max() >= 255:
        return labels
    k = max(5, preset.line_px | 1)
    out = labels.astype(np.uint8)
    for _ in range(2):
        out = cv2.medianBlur(out, k)
    return out.astype(np.int32)


def segment(
    raw_lines: np.ndarray,
    subject: np.ndarray,
    cartoon: np.ndarray,
    preset: DifficultyPreset,
    palette_lab: np.ndarray | None = None,
    strokes: np.ndarray | None = None,
) -> dict[str, Any]:
    """Core algorithm, pure numpy in/out (see module docstring).

    ``raw_lines`` delimit zones; ``strokes`` (subset actually drawn in the
    cartoon) feed detail strokes and ink dots. Defaults to ``raw_lines``.
    """
    if strokes is None:
        strokes = raw_lines
    h, w = subject.shape
    gap = max(1, preset.gap_close_px // 2)
    lines = cv2.morphologyEx(raw_lines.astype(np.uint8), cv2.MORPH_CLOSE, disk(gap)) > 0
    lines = cv2.dilate(lines.astype(np.uint8), disk(1)) > 0
    free = subject & ~lines
    if not free.any():
        raise ValueError("no free pixels left inside the subject: lines cover everything")

    n, zones = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
    n -= 1
    _, (iy, ix) = ndimage.distance_transform_edt(~free, return_indices=True)
    partition = zones[iy, ix].astype(np.int32)
    partition[~subject] = 0

    lab = rgb_to_lab(cartoon)
    flat_free = free.ravel()
    ids = partition.ravel()[flat_free]
    area = np.bincount(partition.ravel(), minlength=n + 1)
    color_w = np.bincount(ids, minlength=n + 1).astype(np.float64)
    color_sum = np.stack(
        [np.bincount(ids, weights=lab[..., c].ravel()[flat_free], minlength=n + 1) for c in range(3)], axis=1
    )
    inner_dt = ndimage.distance_transform_edt(free)
    thickness = np.zeros(n + 1)
    thickness[1:] = ndimage.maximum(inner_dt, labels=zones, index=np.arange(1, n + 1))
    graph = ZoneGraph(n, area, color_sum, color_w, adjacency(partition, n), thickness)
    subject_area = float(subject.sum())
    simplify(graph, preset, subject_area)

    roots = np.array([graph.find(i) for i in range(n + 1)], dtype=np.int32)
    final = roots[partition]
    final[~subject] = 0
    final, old_by_new = _compact(final, {int(u): int(u) for u in np.unique(final)})
    final = smooth_labels(final, preset)
    final, old_by_new = _compact(final, old_by_new)
    uniq = np.arange(1, final.max() + 1)

    borders = border_mask(final)
    zone_lines = cv2.dilate(borders.astype(np.uint8), disk(max(1, preset.line_px // 2))) > 0
    silhouette = border_mask((final > 0).astype(np.int32))
    outline = cv2.dilate(silhouette.astype(np.uint8), disk(max(1, preset.outline_px // 2))) > 0
    final_lines = zone_lines | outline
    details = np.zeros_like(final_lines)
    if preset.keep_detail_strokes:
        details = detail_strokes(strokes, borders, final > 0, preset)
        details = cv2.dilate(details.astype(np.uint8), disk(max(1, preset.detail_px // 2))) > 0
        details &= final > 0
    details |= ink_dots(strokes, borders, final > 0, preset.dark_fill_max_frac * subject_area)

    regions = []
    black_ids = set()
    slices = ndimage.find_objects(final)
    for new_id in range(1, len(uniq) + 1):
        old = old_by_new[new_id]
        is_black = old in graph.black
        if is_black:
            black_ids.add(new_id)
        mean_lab = graph.color(old)
        if palette_lab is not None and len(palette_lab) and not is_black:
            de = delta_e(palette_lab, mean_lab[None, :])
            j = int(np.argmin(de))
            if float(de[j]) <= PHOTO_SNAP_MAX_DE:
                mean_lab = palette_lab[j]
        color_hex = BLACK_HEX if is_black else lab_to_hex(mean_lab)
        name = "noir" if is_black else nearest_crayon(mean_lab)[0]
        sl = slices[new_id - 1]
        ys, xs = np.nonzero(final[sl] == new_id)
        centroid = [float(xs.mean() + sl[1].start), float(ys.mean() + sl[0].start)] if len(xs) else [0.0, 0.0]
        regions.append(
            {
                "id": new_id,
                "area_px": int(len(xs)),
                "color_hex": color_hex,
                "color_name": name,
                "black_fill": is_black,
                "centroid": [round(centroid[0], 1), round(centroid[1], 1)],
            }
        )
    return {
        "labels": final,
        "lines": final_lines | details,
        "outline": outline,
        "details": details,
        "regions": regions,
        "black_ids": black_ids,
        "initial_zone_count": n,
    }


def render(labels: np.ndarray, lines: np.ndarray, regions: list[dict[str, Any]], *, colored: bool, preview: bool = False) -> np.ndarray:
    """Coloring (white zones) or model (colored zones) image with soft strokes."""
    h, w = labels.shape
    lut = np.full((labels.max() + 1, 3), 255, np.uint8)
    rng = np.random.default_rng(7)
    for r in regions:
        if r["black_fill"]:
            lut[r["id"]] = hex_to_rgb(BLACK_HEX)
        elif preview:
            lut[r["id"]] = rng.integers(70, 245, size=3)
        elif colored:
            lut[r["id"]] = hex_to_rgb(r["color_hex"])
    img = lut[labels].astype(np.float32)
    alpha = antialiased(lines)[..., None]
    ink = np.array(hex_to_rgb(BLACK_HEX), np.float32)
    return np.clip(img * (1 - alpha) + ink * alpha + 0.5, 0, 255).astype(np.uint8)


class RegionSegment(BaseTool):
    name = "region_segment"
    version = "1.0.0"
    tier = ToolTier.ANALYZE
    capability = "regions"
    provider = "opendrawer"
    stability = ToolStability.BETA
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    dependencies = ["python:cv2", "python:scipy", "python:skimage"]
    install_instructions = "pip install opencv-python-headless scipy scikit-image"
    capabilities = ["closed_regions", "region_merging", "black_fill_small_details", "photo_color_snapping"]
    best_for = ["zones fermées adaptées à l'âge, couleurs réelles"]

    input_schema = {
        "type": "object",
        "required": ["line_art", "output_dir"],
        "properties": {
            "line_art": {"type": "object", "description": "line_art artifact"},
            "palette": {"type": "array", "description": "photo_analysis.palette (real photo colors)"},
            "color_source": {"type": "string", "enum": ["photo", "cartoon"], "default": "photo"},
            "difficulty": {"type": "string", "description": "Defaults to line_art.difficulty"},
            "output_dir": {"type": "string"},
            "max_regions": {"type": "integer", "description": "Override the preset's maximum"},
            "min_region_frac": {"type": "number", "description": "Override the preset's minimum zone size"},
        },
    }
    resource_profile = ResourceProfile(cpu_cores=2, ram_mb=3000)
    idempotency_key_fields = ["line_art", "palette", "color_source", "difficulty", "max_regions", "min_region_frac"]

    def get_status(self) -> ToolStatus:
        try:
            import scipy  # noqa: F401
            import skimage  # noqa: F401

            return ToolStatus.AVAILABLE
        except ImportError:
            return ToolStatus.UNAVAILABLE

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        started = time.monotonic()
        try:
            line_art = inputs["line_art"]
            preset = get_preset(inputs.get("difficulty") or line_art.get("difficulty"))
            overrides = {k: inputs[k] for k in ("max_regions", "min_region_frac") if inputs.get(k) is not None}
            if overrides:
                from dataclasses import replace

                preset = replace(preset, **overrides)
            raw_lines = ~load_mask(line_art["line_mask_path"])
            strokes = ~load_mask(line_art["strokes_mask_path"]) if line_art.get("strokes_mask_path") else None
            subject = load_mask(line_art["subject_mask_path"])
            cartoon = load_rgb(line_art.get("flat_path") or line_art["source_cartoon"])
            if cartoon.shape[:2] != subject.shape:
                cartoon = resize_to(cartoon, subject.shape[1], subject.shape[0])
        except Exception as exc:
            return ToolResult(success=False, error=f"region_segment: invalid inputs: {exc}")

        color_source = inputs.get("color_source", "photo")
        palette_lab = None
        if color_source == "photo" and inputs.get("palette"):
            palette_lab = rgb_to_lab(np.array([hex_to_rgb(c["hex"]) for c in inputs["palette"]], np.uint8))

        try:
            result = segment(raw_lines, subject, cartoon, preset, palette_lab, strokes)
        except Exception as exc:
            return ToolResult(success=False, error=f"region_segment failed: {exc}")

        out = Path(inputs["output_dir"])
        labels_path = save_rgb(encode_labels(result["labels"]), out / "labels.png")
        lines_path = save_mask(~result["lines"], out / "lines.png")
        coloring = render(result["labels"], result["lines"], result["regions"], colored=False)
        model = render(result["labels"], result["lines"], result["regions"], colored=True)
        preview = render(result["labels"], result["lines"], result["regions"], colored=True, preview=True)
        coloring_path = save_rgb(coloring, out / "coloring_work.png")
        model_path = save_rgb(model, out / "model_work.png")
        preview_path = save_rgb(preview, out / "regions_preview.png")

        regions = result["regions"]
        used = {}
        for r in regions:
            if not r["black_fill"]:
                used.setdefault(r["color_hex"], {"hex": r["color_hex"], "name": r["color_name"]})
        h, w = result["labels"].shape
        region_map = {
            "version": "1.0",
            "difficulty": preset.name,
            "work_size": {"width": w, "height": h},
            "labels_path": str(labels_path),
            "lines_path": str(lines_path),
            "coloring_png": str(coloring_path),
            "model_png": str(model_path),
            "preview_png": str(preview_path),
            "color_source": color_source if palette_lab is not None else "cartoon",
            "region_count": sum(1 for r in regions if not r["black_fill"]),
            "black_fill_count": sum(1 for r in regions if r["black_fill"]),
            "regions": regions,
            "palette_used": list(used.values()),
        }
        return ToolResult(
            success=True,
            data={"region_map": region_map, "initial_zone_count": result["initial_zone_count"]},
            artifacts=[str(p) for p in (labels_path, lines_path, coloring_path, model_path, preview_path)],
            duration_seconds=time.monotonic() - started,
        )
