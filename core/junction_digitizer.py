"""
Junction-aware multi-curve digitizer for scanned B/W / grayscale plots.

Key innovation over naive skeleton+A*:
  - Grayscale intensity is a first-class cue: the cost function, junction
    disambiguation, and anchor snapping all use intensity matching to stay
    on the correct curve through overlaps.
  - HoughLinesP + inpainting removes axes, dashed guidelines, and tick
    marks *before* skeletonisation, preventing false junctions.
  - Branch-aware anchor snapping: K-nearest candidates scored by
    direction consistency + intensity similarity.
  - Stateful Dijkstra with (node, incoming_edge) state: turn-angle^2 +
    intensity-mismatch + darkness + curvature + x-monotonicity cost.
  - Beam search (K candidates per segment) with global crossing check.

Dependencies: opencv-python, numpy, scipy, scikit-image
"""

from __future__ import annotations

import heapq
import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline

logger = logging.getLogger(__name__)

# 8-connectivity offsets (dy, dx)
_NBRS_8 = [(-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)]

# ====================================================================
# Configuration
# ====================================================================

@dataclass
class JunctionConfig:
    """All tunable parameters for the pipeline."""
    # --- Preprocessing ---
    median_ksize: int = 3
    adaptive_block: int = 51
    adaptive_C: int = 10
    morph_close_kernel: Tuple[int, int] = (3, 7)
    grid_h_kernel_ratio: float = 0.20
    grid_v_kernel_ratio: float = 0.20
    hough_line_min_length: int = 80
    hough_line_max_gap: int = 12
    hough_threshold: int = 60
    line_mask_dilate: int = 3
    inpaint_radius: int = 5

    # --- Cost map ---
    cost_w_intensity: float = 1.0
    cost_w_edge: float = 0.5
    cost_w_dist: float = 0.3

    # --- Graph ---
    min_edge_length: int = 5
    pca_window: int = 12

    # --- Pathfinding ---
    lambda_turn: float = 8.0
    alpha_intensity: float = 12.0
    mu_curvature: float = 2.0
    nu_darkness: float = 0.5
    x_decrease_penalty: float = 50.0

    # --- Beam search ---
    beam_width: int = 5
    max_curves: int = 15

    # --- Anchor snapping ---
    snap_radius: int = 30
    snap_k: int = 8

    # --- A* tracing (single-curve mode) ---
    lambda_angle: float = 2.0
    lambda_junction: float = 5.0

    # --- Postprocess ---
    spline_smoothing: float = 0.0
    resample_n: int = 300
    min_curve_xspan_ratio: float = 0.15

    # --- Multi-scale junction refinement ---
    refine_enabled: bool = True
    refine_roi_radius: int = 60
    refine_upscale: int = 4
    refine_boundary_snap: int = 15
    refine_min_edges: int = 3
    refine_extrapolation_len: int = 20

    # --- Debug ---
    debug: bool = False
    debug_dir: str = ""


# ====================================================================
# Data structures
# ====================================================================

@dataclass
class GraphEdge:
    """An edge in the compressed skeleton graph."""
    edge_id: int
    node_a: Tuple[int, int]
    node_b: Tuple[int, int]
    pixels: List[Tuple[int, int]]
    length: int = 0
    curvature: float = 0.0
    x_span: int = 0
    mean_darkness: float = 0.0
    mean_intensity: float = 128.0
    intensity_std: float = 30.0
    mean_edge_strength: float = 0.0
    tangent_a: Optional[np.ndarray] = None
    tangent_b: Optional[np.ndarray] = None

    def __post_init__(self):
        self.length = len(self.pixels)
        if self.pixels:
            xs = [p[0] for p in self.pixels]
            self.x_span = max(xs) - min(xs)

    def other_node(self, node: Tuple[int, int]) -> Tuple[int, int]:
        return self.node_b if node == self.node_a else self.node_a

    def tangent_at(self, node: Tuple[int, int]) -> Optional[np.ndarray]:
        if node == self.node_a:
            return self.tangent_a
        return self.tangent_b


@dataclass
class JunctionPairing:
    """A preferred continuation through a junction node."""
    junction: Tuple[int, int]
    edge_in: int
    edge_out: int
    cos_similarity: float
    intensity_delta: float = 0.0


@dataclass
class AffineMapping:
    """Affine pixel-to-data mapping."""
    matrix_p2d: np.ndarray
    matrix_d2p: np.ndarray
    error_px: float = 0.0


# ====================================================================
# A) PREPROCESS
# ====================================================================

def preprocess(
    image: np.ndarray,
    config: JunctionConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Full preprocessing pipeline.

    Returns (skeleton, binary, gray, cost_map).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    denoised = cv2.medianBlur(gray, config.median_ksize)

    # Detect axes + dashed guide lines via HoughLinesP and remove via inpainting
    binary_for_lines = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, config.adaptive_block, config.adaptive_C,
    )

    # Close dashes to connect them before Hough
    h_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    v_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    connected = binary_for_lines.copy()
    connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, h_close)
    connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, v_close)

    lines = cv2.HoughLinesP(
        connected, 1, np.pi / 180,
        threshold=config.hough_threshold,
        minLineLength=config.hough_line_min_length,
        maxLineGap=config.hough_line_max_gap,
    )

    h, w = gray.shape
    line_mask = np.zeros((h, w), dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            if angle < 8 or angle > 172 or (82 < angle < 98):
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 1)

    if config.line_mask_dilate > 0:
        dilate_k = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (config.line_mask_dilate * 2 + 1, config.line_mask_dilate * 2 + 1),
        )
        line_mask = cv2.dilate(line_mask, dilate_k)

    # Also detect grid lines using long morphological opens
    h_ksize = max(25, int(w * config.grid_h_kernel_ratio))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_ksize, 1))
    h_lines_morph = cv2.morphologyEx(binary_for_lines, cv2.MORPH_OPEN, h_kernel)
    line_mask = cv2.bitwise_or(line_mask, h_lines_morph)

    v_ksize = max(25, int(h * config.grid_v_kernel_ratio))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_ksize))
    v_lines_morph = cv2.morphologyEx(binary_for_lines, cv2.MORPH_OPEN, v_kernel)
    line_mask = cv2.bitwise_or(line_mask, v_lines_morph)

    # Safeguard: if the line mask covers too much of the foreground,
    # it is likely detecting curves rather than axes — skip inpainting.
    fg_total = int(np.sum(binary_for_lines > 0))
    overlap = int(np.sum((line_mask > 0) & (binary_for_lines > 0)))
    if fg_total > 0 and overlap > fg_total * 0.25:
        logger.debug("preprocess: line_mask covers %.0f%% of fg — skipping inpaint",
                      100.0 * overlap / fg_total)
        line_mask[:] = 0

    # Inpaint to remove lines while preserving curve continuity
    if np.any(line_mask > 0):
        cleaned_gray = cv2.inpaint(denoised, line_mask, config.inpaint_radius, cv2.INPAINT_TELEA)
    else:
        cleaned_gray = denoised

    binary = cv2.adaptiveThreshold(
        cleaned_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, config.adaptive_block, config.adaptive_C,
    )

    # Remove text-like connected components
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    plot_area = h * w
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = bw / max(bh, 1)
        x_span_ratio = bw / w

        is_tiny = area < max(30, int(plot_area * 0.0002))
        is_text_like = (
            area < plot_area * 0.008
            and 0.15 < aspect < 6.0
            and x_span_ratio < 0.08
        )
        if is_tiny or is_text_like:
            binary[labels == i] = 0

    # Border cleanup
    border = 3
    binary[:border, :] = 0
    binary[-border:, :] = 0
    binary[:, :border] = 0
    binary[:, -border:] = 0

    # Morphological close to bridge small gaps
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, config.morph_close_kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)

    # Second pass: remove small fragments by x-span
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    min_xspan = max(20, int(w * config.min_curve_xspan_ratio * 0.3))
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_WIDTH] < min_xspan:
            binary[labels == i] = 0

    # Skeletonize
    try:
        skeleton = cv2.ximgproc.thinning(
            binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except AttributeError:
        from skimage.morphology import skeletonize as _skeletonize
        skeleton = (_skeletonize(binary > 0).astype(np.uint8)) * 255

    cost_map = build_cost_map(gray, binary, config)

    return skeleton, binary, gray, cost_map


# ====================================================================
# B) CURVE LIKELIHOOD / COST MAP
# ====================================================================

def build_cost_map(
    gray: np.ndarray,
    binary: np.ndarray,
    config: JunctionConfig,
) -> np.ndarray:
    """Build cost map: low values = likely curve centerline.

    Combines normalised intensity, edge strength, and distance-to-background.
    """
    h, w = gray.shape
    intensity_norm = gray.astype(np.float32) / 255.0

    sx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    sy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    edge_mag = np.sqrt(sx * sx + sy * sy)
    edge_max = edge_mag.max()
    edge_norm = edge_mag / edge_max if edge_max > 0 else np.zeros_like(edge_mag)

    if binary.any():
        dt = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        dt_max = dt.max()
        dt_norm = 1.0 - (dt / dt_max) if dt_max > 0 else np.ones((h, w), dtype=np.float32)
    else:
        dt_norm = np.ones((h, w), dtype=np.float32)

    cost = (
        config.cost_w_intensity * intensity_norm
        + config.cost_w_edge * (1.0 - edge_norm)
        + config.cost_w_dist * dt_norm
    )
    return cost


# ====================================================================
# C) SKELETON GRAPH
# ====================================================================

def _pixel_set_from_skeleton(skeleton: np.ndarray) -> Set[Tuple[int, int]]:
    ys, xs = np.where(skeleton > 0)
    return set(zip(xs.tolist(), ys.tolist()))


def _classify_pixel(pixel: Tuple[int, int], pixel_set: Set[Tuple[int, int]]) -> int:
    x, y = pixel
    return sum(1 for dy, dx in _NBRS_8 if (x + dx, y + dy) in pixel_set)


def build_graph(
    skeleton: np.ndarray,
    gray: np.ndarray,
    cost_map: np.ndarray,
    config: JunctionConfig,
) -> Tuple[
    Dict[Tuple[int, int], List[GraphEdge]],
    List[GraphEdge],
    Set[Tuple[int, int]],
    Set[Tuple[int, int]],
]:
    """Convert skeleton to compressed graph with per-edge intensity features."""
    pixel_set = _pixel_set_from_skeleton(skeleton)
    if not pixel_set:
        return {}, [], set(), set()

    # Classify pixels
    special: Set[Tuple[int, int]] = set()
    regular: Set[Tuple[int, int]] = set()
    for p in pixel_set:
        deg = _classify_pixel(p, pixel_set)
        if deg == 1 or deg >= 3:
            special.add(p)
        else:
            regular.add(p)

    # Merge clusters of junction pixels
    new_endpoints: Set[Tuple[int, int]] = set()
    for p in list(special):
        if p not in pixel_set:
            continue
        deg = _classify_pixel(p, pixel_set)
        if deg >= 3:
            cluster = [p]
            visited_cluster = {p}
            q = [p]
            while q:
                curr = q.pop()
                cx, cy = curr
                for dy, dx in _NBRS_8:
                    nb = (cx + dx, cy + dy)
                    if nb in special and nb not in visited_cluster:
                        if _classify_pixel(nb, pixel_set) >= 3:
                            visited_cluster.add(nb)
                            cluster.append(nb)
                            q.append(nb)
            if len(cluster) > 1:
                cx_mean = int(round(np.mean([c[0] for c in cluster])))
                cy_mean = int(round(np.mean([c[1] for c in cluster])))
                centroid = (cx_mean, cy_mean)
                for c in cluster:
                    if c != centroid:
                        # Keep in pixel_set for connectivity, just demote to regular
                        special.discard(c)
                        regular.add(c)
                if centroid not in pixel_set:
                    pixel_set.add(centroid)
                special.add(centroid)
        elif deg == 0:
            new_endpoints.add(p)

    special |= new_endpoints

    # Re-classify
    endpoints: Set[Tuple[int, int]] = set()
    junctions: Set[Tuple[int, int]] = set()
    for p in special:
        deg = _classify_pixel(p, pixel_set)
        if deg == 1:
            endpoints.add(p)
        elif deg >= 3:
            junctions.add(p)
        elif deg == 0:
            endpoints.add(p)
        else:
            regular.add(p)

    # Compute edge strength image
    sx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    sy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    edge_strength = np.sqrt(sx * sx + sy * sy)
    es_max = edge_strength.max()
    if es_max > 0:
        edge_strength /= es_max

    all_special = endpoints | junctions

    # Trace edges
    all_edges: List[GraphEdge] = []
    edge_id_counter = 0
    visited_starts: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()

    for start in all_special:
        sx_s, sy_s = start
        for dy, dx in _NBRS_8:
            nb = (sx_s + dx, sy_s + dy)
            if nb not in pixel_set:
                continue
            if nb in all_special:
                key_fwd = (start, nb)
                key_rev = (nb, start)
                if key_fwd in visited_starts or key_rev in visited_starts:
                    continue
                visited_starts.add(key_fwd)
                edge = _make_edge(edge_id_counter, start, nb, [start, nb],
                                  gray, edge_strength, config)
                edge_id_counter += 1
                all_edges.append(edge)
                continue

            key_check = (start, nb)
            if key_check in visited_starts:
                continue

            chain = [start, nb]
            visited_chain = {start, nb}
            current = nb
            found_end = False

            while True:
                cx, cy = current
                next_pixel = None
                for ddy, ddx in _NBRS_8:
                    candidate = (cx + ddx, cy + ddy)
                    if candidate not in pixel_set or candidate in visited_chain:
                        continue
                    if candidate in all_special:
                        chain.append(candidate)
                        visited_starts.add((start, nb))
                        visited_starts.add((candidate, chain[-2]))
                        edge = _make_edge(edge_id_counter, start, candidate, chain,
                                          gray, edge_strength, config)
                        edge_id_counter += 1
                        all_edges.append(edge)
                        found_end = True
                        break
                    next_pixel = candidate
                    break

                if found_end:
                    break
                if next_pixel is None:
                    break
                chain.append(next_pixel)
                visited_chain.add(next_pixel)
                current = next_pixel

    # Spur removal
    filtered = []
    for e in all_edges:
        if e.length < config.min_edge_length:
            if (e.node_a in endpoints and e.node_b in junctions) or \
               (e.node_b in endpoints and e.node_a in junctions):
                continue
        filtered.append(e)
    all_edges = filtered

    adjacency: Dict[Tuple[int, int], List[GraphEdge]] = {}
    for e in all_edges:
        adjacency.setdefault(e.node_a, []).append(e)
        adjacency.setdefault(e.node_b, []).append(e)

    logger.info("build_graph: %d edges, %d endpoints, %d junctions",
                len(all_edges), len(endpoints), len(junctions))
    return adjacency, all_edges, endpoints, junctions


def _make_edge(
    edge_id: int,
    node_a: Tuple[int, int],
    node_b: Tuple[int, int],
    pixels: List[Tuple[int, int]],
    gray: np.ndarray,
    edge_strength: np.ndarray,
    config: JunctionConfig,
) -> GraphEdge:
    h, w = gray.shape
    intensities = []
    es_vals = []
    for px, py in pixels:
        if 0 <= py < h and 0 <= px < w:
            intensities.append(float(gray[py, px]))
            es_vals.append(float(edge_strength[py, px]))

    mean_intensity = float(np.mean(intensities)) if intensities else 128.0
    intensity_std = float(np.std(intensities)) if len(intensities) > 1 else 30.0
    mean_es = float(np.mean(es_vals)) if es_vals else 0.0
    curvature = _compute_curvature(pixels)
    tangent_a = _compute_tangent(pixels, True, config.pca_window)
    tangent_b = _compute_tangent(pixels, False, config.pca_window)

    return GraphEdge(
        edge_id=edge_id, node_a=node_a, node_b=node_b, pixels=pixels,
        curvature=curvature, mean_darkness=mean_intensity,
        mean_intensity=mean_intensity, intensity_std=intensity_std,
        mean_edge_strength=mean_es, tangent_a=tangent_a, tangent_b=tangent_b,
    )


def _compute_curvature(pixels: List[Tuple[int, int]]) -> float:
    if len(pixels) < 3:
        return 0.0
    total = 0.0
    step = max(1, len(pixels) // 20)
    for i in range(step, len(pixels) - step, step):
        x0, y0 = pixels[i - step]
        x1, y1 = pixels[i]
        x2, y2 = pixels[i + step]
        v1 = (x1 - x0, y1 - y0)
        v2 = (x2 - x1, y2 - y1)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        total += abs(math.atan2(cross, dot))
    return total


def _compute_tangent(
    pixels: List[Tuple[int, int]],
    outward_from_start: bool,
    window: int = 12,
) -> Optional[np.ndarray]:
    if len(pixels) < 3:
        return None
    subset = pixels[:min(window, len(pixels))] if outward_from_start \
        else pixels[max(0, len(pixels) - window):]
    arr = np.array(subset, dtype=np.float64)
    centered = arr - arr.mean(axis=0)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]
    except np.linalg.LinAlgError:
        return None

    endpoint = np.array(subset[0] if outward_from_start else subset[-1], dtype=np.float64)
    far_point = np.array(subset[-1] if outward_from_start else subset[0], dtype=np.float64)
    if np.dot(direction, far_point - endpoint) < 0:
        direction = -direction
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 1e-8 else None


# ====================================================================
# D) JUNCTION DISAMBIGUATION (intensity-aware)
# ====================================================================

def disambiguate_junctions(
    adjacency: Dict[Tuple[int, int], List[GraphEdge]],
    junctions: Set[Tuple[int, int]],
    config: JunctionConfig,
) -> Dict[Tuple[int, int], List[JunctionPairing]]:
    """Pair edges at each junction using tangent angle + intensity consistency.

    score = cos_similarity - alpha * |intensity_in - intensity_out| / 255
    """
    pairings: Dict[Tuple[int, int], List[JunctionPairing]] = {}

    for junc in junctions:
        edges = adjacency.get(junc, [])
        if len(edges) < 2:
            continue

        tangents = [(e.edge_id, e.tangent_at(junc)) for e in edges]

        pair_scores: List[Tuple[float, int, int, float, float]] = []
        for i in range(len(tangents)):
            for j in range(i + 1, len(tangents)):
                eid_i, t_i = tangents[i]
                eid_j, t_j = tangents[j]
                cos_sim = float(-np.dot(t_i, t_j)) if t_i is not None and t_j is not None else 0.0

                e_i = next(e for e in edges if e.edge_id == eid_i)
                e_j = next(e for e in edges if e.edge_id == eid_j)
                intensity_delta = abs(e_i.mean_intensity - e_j.mean_intensity)
                score = cos_sim - config.alpha_intensity * (intensity_delta / 255.0)
                pair_scores.append((score, eid_i, eid_j, cos_sim, intensity_delta))

        pair_scores.sort(key=lambda x: -x[0])
        used: Set[int] = set()
        junc_pairings: List[JunctionPairing] = []

        for score, eid_i, eid_j, cos_sim, i_delta in pair_scores:
            if eid_i in used or eid_j in used:
                continue
            junc_pairings.append(JunctionPairing(
                junction=junc, edge_in=eid_i, edge_out=eid_j,
                cos_similarity=cos_sim, intensity_delta=i_delta,
            ))
            junc_pairings.append(JunctionPairing(
                junction=junc, edge_in=eid_j, edge_out=eid_i,
                cos_similarity=cos_sim, intensity_delta=i_delta,
            ))
            used.add(eid_i)
            used.add(eid_j)

            logger.debug(
                "junction (%d,%d): pair e%d<->e%d cos=%.2f di=%.1f score=%.2f",
                junc[0], junc[1], eid_i, eid_j, cos_sim, i_delta, score,
            )

        pairings[junc] = junc_pairings

    logger.info("disambiguate_junctions: paired %d junctions", len(pairings))
    return pairings


# ====================================================================
# D1) ANCHOR-GUIDED DIRECT PIXEL TRACING (single-curve mode)
# ====================================================================

def _astar_on_skeleton(
    skeleton: np.ndarray,
    gray: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    ref_intensity: float,
    max_off_skeleton: int = 3,
    junctions: Optional[Set[Tuple[int, int]]] = None,
    lambda_angle: float = 2.0,
    lambda_junction: float = 5.0,
) -> Optional[List[Tuple[int, int]]]:
    """A* search on skeleton pixels with turn-awareness and junction penalty.

    Turn-aware: penalises sharp direction changes to prevent the trace
    from switching to another curve's skeleton branch at junctions.
    Junction penalty: extra cost for traversing junction pixels.
    """
    h, w = skeleton.shape
    sx, sy = start
    gx, gy = goal

    if not (0 <= sy < h and 0 <= sx < w and 0 <= gy < h and 0 <= gx < w):
        return None

    junction_set = junctions or set()

    def heuristic(x: int, y: int) -> float:
        return math.hypot(gx - x, gy - y)

    # Priority queue: (f_cost, counter, x, y, off_skel_run)
    pq: List[Tuple[float, int, int, int, int]] = [(heuristic(sx, sy), 0, sx, sy, 0)]
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], float] = {(sx, sy): 0.0}
    counter = 0

    # Initial direction: toward goal
    init_dx = float(gx - sx)
    init_dy = float(gy - sy)
    init_norm = math.hypot(init_dx, init_dy)
    if init_norm > 0:
        init_dx /= init_norm
        init_dy /= init_norm

    while pq:
        _, _, cx, cy, off_run = heapq.heappop(pq)

        if (cx, cy) == (gx, gy):
            path = [(gx, gy)]
            curr = (gx, gy)
            while curr in came_from:
                curr = came_from[curr]
                path.append(curr)
            path.reverse()
            return path

        cur_g = g_score.get((cx, cy), float("inf"))

        # Incoming direction from came_from (or initial direction)
        if (cx, cy) in came_from:
            px, py = came_from[(cx, cy)]
            in_dx, in_dy = float(cx - px), float(cy - py)
        else:
            in_dx, in_dy = init_dx, init_dy
        in_norm = math.hypot(in_dx, in_dy)
        if in_norm > 0:
            in_dx /= in_norm
            in_dy /= in_norm

        for dy, dx in _NBRS_8:
            nx, ny = cx + dx, cy + dy
            if not (0 <= ny < h and 0 <= nx < w):
                continue

            on_skel = skeleton[ny, nx] > 0
            new_off_run = 0 if on_skel else off_run + 1
            if new_off_run > max_off_skeleton:
                continue

            # Movement cost
            step = 1.414 if abs(dx) + abs(dy) == 2 else 1.0
            off_penalty = 0.0 if on_skel else 8.0
            pix_intensity = float(gray[ny, nx])
            intensity_pen = abs(pix_intensity - ref_intensity) / 255.0 * 2.0
            darkness_pen = pix_intensity / 255.0 * 0.5

            # Turn angle penalty — discourages sharp direction changes
            out_norm = math.hypot(float(dx), float(dy))
            if out_norm > 0 and in_norm > 0:
                cos_a = (in_dx * dx + in_dy * dy) / out_norm
                cos_a = max(-1.0, min(1.0, cos_a))
                turn = math.acos(cos_a)  # 0..pi
                turn_pen = lambda_angle * (turn / math.pi) ** 2
            else:
                turn_pen = 0.0

            # Junction penalty — discourages traversing junction pixels
            junc_pen = lambda_junction if (nx, ny) in junction_set else 0.0

            move_cost = (step + off_penalty + intensity_pen + darkness_pen
                         + turn_pen + junc_pen)
            tentative = cur_g + move_cost

            if tentative < g_score.get((nx, ny), float("inf")):
                g_score[(nx, ny)] = tentative
                came_from[(nx, ny)] = (cx, cy)
                f = tentative + heuristic(nx, ny)
                counter += 1
                heapq.heappush(pq, (f, counter, nx, ny, new_off_run))

    return None


def _snap_to_skeleton(
    point: Tuple[int, int],
    skeleton: np.ndarray,
    radius: int = 30,
) -> Tuple[int, int]:
    """Find the closest skeleton pixel to *point* within *radius*."""
    px, py = point
    h, w = skeleton.shape
    x0 = max(0, px - radius)
    y0 = max(0, py - radius)
    x1 = min(w, px + radius + 1)
    y1 = min(h, py + radius + 1)

    roi = skeleton[y0:y1, x0:x1]
    ys, xs = np.where(roi > 0)
    if len(xs) == 0:
        return point  # no skeleton nearby, return as-is
    xs += x0
    ys += y0
    dists = (xs - px) ** 2 + (ys - py) ** 2
    idx = int(np.argmin(dists))
    return (int(xs[idx]), int(ys[idx]))


def select_anchors_interactive(
    image: np.ndarray,
    title: str = "Click anchor points on the curve, then close the window",
) -> List[Tuple[int, int]]:
    """Display image and let user click anchor points via matplotlib ginput.

    Left-click to add anchors sequentially along the curve.
    Right-click or press Enter to finish.
    Returns a list of (x, y) pixel coordinates.
    """
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    if len(image.shape) == 2:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.set_xlabel("Left-click to add anchor | Right-click or Enter to finish")

    points = plt.ginput(n=-1, timeout=0)
    plt.close(fig)

    anchors = [(int(round(x)), int(round(y))) for x, y in points]
    logger.info("select_anchors_interactive: %d anchors selected", len(anchors))
    return anchors


def trace_single_curve_anchored(
    anchors: List[Tuple[int, int]],
    skeleton: np.ndarray,
    gray: np.ndarray,
    config: JunctionConfig,
    junctions: Optional[Set[Tuple[int, int]]] = None,
) -> Optional[List[Tuple[int, int]]]:
    """Trace ONE curve by walking along skeleton pixels through ordered anchors.

    This completely bypasses the graph/junction machinery.  The anchors
    constrain the trace so it cannot swap to another curve at overlaps.

    Steps:
      1. Snap each anchor to the nearest skeleton pixel.
      2. Estimate the curve's intensity from the anchor neighbourhood.
      3. A* between consecutive snapped anchors on the skeleton, using
         intensity matching as a cost term.
      4. Concatenate segments into one ordered pixel list.
      5. Extend leftward and rightward from the first/last anchor along
         the skeleton to capture the full curve extent.
    """
    if len(anchors) < 2:
        logger.warning("trace_single_curve_anchored: need >= 2 anchors")
        return None

    h, w = skeleton.shape

    # 1. Snap anchors to skeleton
    snapped = [_snap_to_skeleton(a, skeleton, config.snap_radius) for a in anchors]

    # 2. Estimate reference intensity (darkest quartile near anchors)
    intensities = []
    for ax, ay in snapped:
        r = 5
        patch = gray[max(0, ay - r):min(h, ay + r + 1),
                      max(0, ax - r):min(w, ax + r + 1)]
        if patch.size > 0:
            sorted_v = np.sort(patch.ravel())
            q = max(1, len(sorted_v) // 4)
            intensities.append(float(np.mean(sorted_v[:q])))
    ref_intensity = float(np.median(intensities)) if intensities else 60.0

    logger.info("trace_single_curve_anchored: %d anchors, ref_intensity=%.0f",
                len(anchors), ref_intensity)

    # 3. A* between consecutive anchors
    full_path: List[Tuple[int, int]] = []
    for i in range(len(snapped) - 1):
        segment = _astar_on_skeleton(
            skeleton, gray, snapped[i], snapped[i + 1],
            ref_intensity, max_off_skeleton=5,
            junctions=junctions,
            lambda_angle=config.lambda_angle,
            lambda_junction=config.lambda_junction)
        if segment is None:
            logger.warning("trace_single_curve_anchored: A* failed %d->%d, "
                           "falling back to straight line", i, i + 1)
            segment = [snapped[i], snapped[i + 1]]

        # Avoid duplicating the junction point
        start_idx = 1 if full_path and segment and segment[0] == full_path[-1] else 0
        full_path.extend(segment[start_idx:])

    if not full_path:
        return None

    # 4. Extend curve beyond first and last anchor
    full_path = _extend_along_skeleton(full_path, skeleton, gray,
                                        ref_intensity, extend_left=True)
    full_path = _extend_along_skeleton(full_path, skeleton, gray,
                                        ref_intensity, extend_left=False)

    logger.info("trace_single_curve_anchored: traced %d pixels", len(full_path))
    return full_path


def _extend_along_skeleton(
    path: List[Tuple[int, int]],
    skeleton: np.ndarray,
    gray: np.ndarray,
    ref_intensity: float,
    extend_left: bool,
    max_extension: int = 500,
    intensity_tolerance: float = 60.0,
) -> List[Tuple[int, int]]:
    """Walk along the skeleton beyond the path endpoint to capture the
    full extent of the curve, as long as the intensity stays consistent."""
    h, w = skeleton.shape

    if extend_left:
        path = list(reversed(path))

    # Compute incoming direction from last few pixels
    if len(path) < 3:
        return list(reversed(path)) if extend_left else path

    tip = path[-1]
    prev = path[-min(8, len(path))]
    direction = np.array([tip[0] - prev[0], tip[1] - prev[1]], dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction /= norm

    visited = set(map(tuple, path[-50:]))  # recent pixels
    extension: List[Tuple[int, int]] = []
    current = tip

    for _ in range(max_extension):
        cx, cy = current
        best_score = float("inf")
        best_nb = None

        for dy, dx in _NBRS_8:
            nx, ny = cx + dx, cy + dy
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if (nx, ny) in visited:
                continue
            if skeleton[ny, nx] == 0:
                continue

            # Score: prefer continuing in same direction + matching intensity
            mv = np.array([nx - cx, ny - cy], dtype=np.float64)
            mv_norm = np.linalg.norm(mv)
            if mv_norm > 0:
                mv /= mv_norm
            dir_score = 1.0 - float(np.dot(direction, mv))  # 0 = same dir, 2 = opposite
            pix_int = float(gray[ny, nx])
            int_score = abs(pix_int - ref_intensity) / 255.0

            score = dir_score * 2.0 + int_score
            if score < best_score:
                best_score = score
                best_nb = (nx, ny)

        if best_nb is None:
            break

        # Check intensity deviation isn't too large
        pix_int = float(gray[best_nb[1], best_nb[0]])
        if abs(pix_int - ref_intensity) > intensity_tolerance:
            break

        extension.append(best_nb)
        visited.add(best_nb)

        # Update direction (smooth)
        old_dir = direction.copy()
        new_mv = np.array([best_nb[0] - current[0], best_nb[1] - current[1]],
                          dtype=np.float64)
        new_norm = np.linalg.norm(new_mv)
        if new_norm > 0:
            direction = 0.7 * old_dir + 0.3 * (new_mv / new_norm)
            d_norm = np.linalg.norm(direction)
            if d_norm > 0:
                direction /= d_norm

        current = best_nb

    result = path + extension
    if extend_left:
        result = list(reversed(result))
    return result


# ====================================================================
# D2) MULTI-SCALE ZOOM JUNCTION REFINEMENT
# ====================================================================

def _build_zoomed_graph(
    original_gray: np.ndarray,
    x0: int, y0: int, x1: int, y1: int,
    scale: int,
    config: JunctionConfig,
) -> Optional[Tuple[
    np.ndarray,  # roi_upscaled
    np.ndarray,  # roi_skeleton
    np.ndarray,  # roi_binary
    Dict[Tuple[int, int], List[GraphEdge]],  # adjacency
    List[GraphEdge],  # edges
    Set[Tuple[int, int]],  # endpoints
    Set[Tuple[int, int]],  # junctions
]]:
    """Crop, upscale, threshold, skeletonize, and build graph on a zoomed ROI."""
    roi = original_gray[y0:y1, x0:x1]
    if roi.size < 100:
        return None

    roi_up = cv2.resize(roi, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_CUBIC)

    denoised = cv2.medianBlur(roi_up, config.median_ksize)

    # Adaptive block size for zoomed image (must be odd, < min dimension)
    min_dim = min(roi_up.shape[:2])
    blk = config.adaptive_block
    if blk >= min_dim:
        blk = max(3, min_dim - 2)
    if blk % 2 == 0:
        blk += 1

    roi_binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blk, config.adaptive_C,
    )

    # Clean border
    border = max(2, scale)
    roi_binary[:border, :] = 0
    roi_binary[-border:, :] = 0
    roi_binary[:, :border] = 0
    roi_binary[:, -border:] = 0

    # Morphological close
    ck = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    roi_binary = cv2.morphologyEx(roi_binary, cv2.MORPH_CLOSE, ck)

    # Remove tiny connected components
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi_binary)
    min_area = scale * scale * 3
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            roi_binary[labels == i] = 0

    # Skeletonize
    try:
        roi_skel = cv2.ximgproc.thinning(
            roi_binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except AttributeError:
        from skimage.morphology import skeletonize as _skel
        roi_skel = (_skel(roi_binary > 0).astype(np.uint8)) * 255

    if np.sum(roi_skel > 0) < 10:
        return None

    roi_cost = build_cost_map(roi_up, roi_binary, config)

    # Graph with relaxed min_edge_length for zoomed view
    from copy import copy as _shallow_copy
    zoom_cfg = _shallow_copy(config)
    zoom_cfg.min_edge_length = max(2, config.min_edge_length // 2)

    roi_adj, roi_edges, roi_eps, roi_juncs = build_graph(
        roi_skel, roi_up, roi_cost, zoom_cfg)

    if not roi_adj:
        return None

    return roi_up, roi_skel, roi_binary, roi_adj, roi_edges, roi_eps, roi_juncs


def _dijkstra_local(
    adjacency: Dict[Tuple[int, int], List[GraphEdge]],
    start: Tuple[int, int],
    end: Tuple[int, int],
    max_steps: int = 8000,
) -> Optional[Tuple[float, List[int]]]:
    """Simple Dijkstra on edge length.  Returns (total_length, edge_id_list)."""
    if start == end:
        return (0.0, [])

    pq: List[Tuple[float, int, Tuple[int, int], Tuple[int, ...]]] = [
        (0.0, 0, start, ())
    ]
    visited: Set[Tuple[int, int]] = set()
    counter = 0

    while pq:
        cost, _, node, eids = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)

        if node == end:
            return (cost, list(eids))

        counter += 1
        if counter > max_steps:
            return None

        for edge in adjacency.get(node, []):
            nxt = edge.other_node(node)
            if nxt not in visited:
                counter += 1
                heapq.heappush(
                    pq, (cost + edge.length, counter, nxt, eids + (edge.edge_id,)))

    return None


def _collect_edge_entry_point(
    edge: GraphEdge,
    junction: Tuple[int, int],
    target_distance: float,
) -> Optional[Tuple[int, int]]:
    """Walk along *edge* pixels starting at *junction* and return the pixel
    that is approximately *target_distance* away from the junction."""
    if edge.node_a == junction:
        pxs = edge.pixels
    elif edge.node_b == junction:
        pxs = list(reversed(edge.pixels))
    else:
        return None
    if not pxs:
        return None

    jx, jy = junction
    best: Tuple[int, int] = pxs[0]
    for px, py in pxs:
        d = math.hypot(px - jx, py - jy)
        best = (px, py)
        if d >= target_distance:
            break
    return best


def _path_edges_to_pixels(
    edge_ids: List[int],
    edge_map: Dict[int, GraphEdge],
    start_node: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """Collect ordered pixels along a sequence of edge IDs."""
    path: List[Tuple[int, int]] = []
    current = start_node
    for eid in edge_ids:
        edge = edge_map.get(eid)
        if edge is None:
            continue
        if edge.node_a == current:
            pixels = edge.pixels
        elif edge.node_b == current:
            pixels = list(reversed(edge.pixels))
        else:
            da = abs(edge.node_a[0] - current[0]) + abs(edge.node_a[1] - current[1])
            db = abs(edge.node_b[0] - current[0]) + abs(edge.node_b[1] - current[1])
            pixels = edge.pixels if da <= db else list(reversed(edge.pixels))
        skip = 1 if path and pixels and pixels[0] == path[-1] else 0
        path.extend(pixels[skip:])
        current = pixels[-1] if pixels else current
    return path


def _snap_to_graph_node(
    zx: float, zy: float,
    all_nodes: List[Tuple[int, int]],
    node_arr: np.ndarray,
    snap_limit: float,
) -> Optional[Tuple[int, int]]:
    """Find the closest graph node to (zx, zy) within *snap_limit*."""
    dists = np.linalg.norm(node_arr - np.array([zx, zy], dtype=np.float64), axis=1)
    idx = int(np.argmin(dists))
    if dists[idx] > snap_limit:
        return None
    return all_nodes[idx]


def refine_single_junction(
    junction: Tuple[int, int],
    edges: List[GraphEdge],
    original_gray: np.ndarray,
    config: JunctionConfig,
) -> Optional[List[JunctionPairing]]:
    """Zoom into a junction region, re-skeletonize at high resolution, and
    determine correct edge pairings by tracing paths in the zoomed view.

    At higher resolution, overlapping curves that appeared merged at global
    resolution may separate into distinct strands, directly revealing which
    global edges belong to the same curve.
    """
    jx, jy = junction
    h, w = original_gray.shape
    r = config.refine_roi_radius

    x0, y0 = max(0, jx - r), max(0, jy - r)
    x1, y1 = min(w, jx + r), min(h, jy + r)
    scale = config.refine_upscale

    zoomed = _build_zoomed_graph(original_gray, x0, y0, x1, y1, scale, config)
    if zoomed is None:
        return None
    roi_up, roi_skel, roi_binary, roi_adj, roi_edges, roi_eps, roi_juncs = zoomed

    all_nodes = list(roi_adj.keys())
    node_arr = np.array(all_nodes, dtype=np.float64)
    snap_limit = config.refine_boundary_snap * scale

    # Map each global edge's entry point to the zoomed graph
    entry_distance = r * 0.6
    entries: List[Dict[str, Any]] = []

    for edge in edges:
        pt = _collect_edge_entry_point(edge, junction, entry_distance)
        if pt is None:
            continue
        zx = (pt[0] - x0) * scale
        zy = (pt[1] - y0) * scale
        zx = max(0.0, min(float(roi_up.shape[1] - 1), zx))
        zy = max(0.0, min(float(roi_up.shape[0] - 1), zy))

        znode = _snap_to_graph_node(zx, zy, all_nodes, node_arr, snap_limit)
        if znode is None:
            continue

        entries.append({
            "edge_id": edge.edge_id,
            "zoomed_node": znode,
            "tangent": edge.tangent_at(junction),
            "intensity": edge.mean_intensity,
            "original_entry": pt,
        })

    if len(entries) < 2:
        return None

    # Score every pair of entry points
    pair_scores: List[Tuple[float, int, int, float, float]] = []
    jpt = np.array(junction, dtype=np.float64)

    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            ei, ej = entries[i], entries[j]

            # Connectivity in zoomed graph
            if ei["zoomed_node"] == ej["zoomed_node"]:
                conn_cost = 0.0
            else:
                res = _dijkstra_local(
                    roi_adj, ei["zoomed_node"], ej["zoomed_node"])
                if res is None:
                    conn_cost = 1e4  # not connected → strongly disfavour
                else:
                    conn_cost = res[0]

            # Tangent alignment (tangents at junction should oppose each other)
            t_i, t_j = ei["tangent"], ej["tangent"]
            cos_sim = float(-np.dot(t_i, t_j)) if (
                t_i is not None and t_j is not None) else 0.0

            # Intensity similarity
            i_delta = abs(ei["intensity"] - ej["intensity"])

            # Spatial alignment: entry points on opposite sides of junction
            v_i = np.array(ei["original_entry"], dtype=np.float64) - jpt
            v_j = np.array(ej["original_entry"], dtype=np.float64) - jpt
            n_i, n_j = np.linalg.norm(v_i), np.linalg.norm(v_j)
            spatial = float(-np.dot(v_i / n_i, v_j / n_j)) if (
                n_i > 1e-8 and n_j > 1e-8) else 0.0

            # Curvature: compute curvature of zoomed path (lower = better)
            curvature_cost = 0.0
            if conn_cost < 1e4 and ei["zoomed_node"] != ej["zoomed_node"]:
                res2 = _dijkstra_local(
                    roi_adj, ei["zoomed_node"], ej["zoomed_node"])
                if res2 is not None:
                    z_edge_map = {e.edge_id: e for e in roi_edges}
                    px_path = _path_edges_to_pixels(
                        res2[1], z_edge_map, ei["zoomed_node"])
                    if len(px_path) >= 5:
                        curvature_cost = _compute_curvature(px_path)

            score = (
                conn_cost * 1.0
                - cos_sim * 40.0
                + i_delta * 0.3
                - spatial * 20.0
                + curvature_cost * config.mu_curvature
            )
            pair_scores.append((
                score, ei["edge_id"], ej["edge_id"], cos_sim, i_delta))

    if not pair_scores:
        return None

    # Greedy pairing — best (lowest) score first
    pair_scores.sort(key=lambda x: x[0])
    used: Set[int] = set()
    pairings: List[JunctionPairing] = []

    for score, eid_i, eid_j, cos_sim, i_delta in pair_scores:
        if eid_i in used or eid_j in used:
            continue
        pairings.append(JunctionPairing(
            junction=junction, edge_in=eid_i, edge_out=eid_j,
            cos_similarity=cos_sim, intensity_delta=i_delta))
        pairings.append(JunctionPairing(
            junction=junction, edge_in=eid_j, edge_out=eid_i,
            cos_similarity=cos_sim, intensity_delta=i_delta))
        used.add(eid_i)
        used.add(eid_j)
        logger.debug(
            "zoom-refine junc (%d,%d): pair e%d<->e%d cos=%.2f di=%.1f "
            "score=%.2f",
            junction[0], junction[1], eid_i, eid_j, cos_sim, i_delta, score)

    return pairings if pairings else None


def refine_junctions_multiscale(
    adjacency: Dict[Tuple[int, int], List[GraphEdge]],
    all_edges: List[GraphEdge],
    junctions: Set[Tuple[int, int]],
    junction_pairings: Dict[Tuple[int, int], List[JunctionPairing]],
    original_gray: np.ndarray,
    config: JunctionConfig,
) -> Tuple[Dict[Tuple[int, int], List[JunctionPairing]], int]:
    """Improve junction pairings via multi-scale zoom for every junction
    with >= *refine_min_edges* incident edges.

    Returns (refined_pairings_dict, number_of_junctions_refined).
    """
    refined = dict(junction_pairings)
    num_refined = 0

    for junc in junctions:
        edges = adjacency.get(junc, [])
        if len(edges) < config.refine_min_edges:
            continue
        result = refine_single_junction(junc, edges, original_gray, config)
        if result is not None:
            refined[junc] = result
            num_refined += 1
            logger.info(
                "Refined junction (%d,%d): %d edges → %d pairings",
                junc[0], junc[1], len(edges), len(result) // 2)

    logger.info("refine_junctions_multiscale: refined %d / %d junctions",
                num_refined, len(junctions))
    return refined, num_refined


# --------------- post-tracing junction pixel replacement ---------------

def refine_curve_pixels_at_junctions(
    curve_pixels: List[Tuple[int, int]],
    junctions: Set[Tuple[int, int]],
    original_gray: np.ndarray,
    config: JunctionConfig,
) -> List[Tuple[int, int]]:
    """After global tracing, zoom into each junction the curve passes through
    and replace the junction-region pixel segment with a re-traced path from
    the zoomed skeleton graph, yielding smoother and more accurate pixels."""
    if not curve_pixels or not junctions:
        return curve_pixels

    h, w = original_gray.shape
    r = config.refine_roi_radius
    half_r = r // 2  # inner region used for pixel replacement
    scale = config.refine_upscale
    snap_limit = config.refine_boundary_snap * scale

    curve_arr = np.array(curve_pixels, dtype=np.float64)

    # Identify junctions the curve passes near
    junc_list = []
    for junc in junctions:
        dists = np.linalg.norm(curve_arr - np.array(junc, dtype=np.float64), axis=1)
        if dists.min() <= 8:
            junc_list.append(junc)

    if not junc_list:
        return curve_pixels

    result = list(curve_pixels)

    for junc in junc_list:
        jx, jy = junc
        roi_x0, roi_y0 = max(0, jx - half_r), max(0, jy - half_r)
        roi_x1, roi_y1 = min(w, jx + half_r), min(h, jy + half_r)

        # Find which indices of *result* fall inside the ROI
        in_roi = [
            idx for idx, (px, py) in enumerate(result)
            if roi_x0 <= px < roi_x1 and roi_y0 <= py < roi_y1
        ]
        if len(in_roi) < 3:
            continue

        entry_idx, exit_idx = in_roi[0], in_roi[-1]
        entry_pt = result[entry_idx]
        exit_pt = result[exit_idx]

        # Build zoomed graph on a slightly larger ROI for context
        bx0, by0 = max(0, jx - r), max(0, jy - r)
        bx1, by1 = min(w, jx + r), min(h, jy + r)

        zoomed = _build_zoomed_graph(
            original_gray, bx0, by0, bx1, by1, scale, config)
        if zoomed is None:
            continue
        roi_up, _, _, roi_adj, roi_edges, _, _ = zoomed

        all_nodes = list(roi_adj.keys())
        if not all_nodes:
            continue
        node_arr = np.array(all_nodes, dtype=np.float64)

        # Map entry/exit to zoomed coords and snap
        z_entry = _snap_to_graph_node(
            (entry_pt[0] - bx0) * scale, (entry_pt[1] - by0) * scale,
            all_nodes, node_arr, snap_limit)
        z_exit = _snap_to_graph_node(
            (exit_pt[0] - bx0) * scale, (exit_pt[1] - by0) * scale,
            all_nodes, node_arr, snap_limit)

        if z_entry is None or z_exit is None or z_entry == z_exit:
            continue

        path_res = _dijkstra_local(roi_adj, z_entry, z_exit)
        if path_res is None:
            continue

        _, path_eids = path_res
        z_edge_map = {e.edge_id: e for e in roi_edges}
        zoomed_px = _path_edges_to_pixels(path_eids, z_edge_map, z_entry)
        if len(zoomed_px) < 3:
            continue

        # Map back to original coordinates
        refined_px = []
        seen: Set[Tuple[int, int]] = set()
        for zx, zy in zoomed_px:
            ox = round(zx / scale + bx0)
            oy = round(zy / scale + by0)
            p = (ox, oy)
            if p not in seen:
                refined_px.append(p)
                seen.add(p)

        if len(refined_px) < 2:
            continue

        # Replace segment while preserving boundary pixels for smooth stitching
        result = result[:entry_idx] + refined_px + result[exit_idx + 1:]

    return result


# --------------- debug helpers for refinement ---------------

def save_debug_junction_zoom(
    image: np.ndarray,
    junctions: Set[Tuple[int, int]],
    adjacency: Dict[Tuple[int, int], List[GraphEdge]],
    config: JunctionConfig,
    output_dir: str,
) -> None:
    """Save zoomed junction ROI images with overlaid skeleton for inspection."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    r = config.refine_roi_radius
    scale = config.refine_upscale

    for idx, junc in enumerate(sorted(junctions)):
        edges = adjacency.get(junc, [])
        if len(edges) < config.refine_min_edges:
            continue

        jx, jy = junc
        x0, y0 = max(0, jx - r), max(0, jy - r)
        x1, y1 = min(w, jx + r), min(h, jy + r)

        zoomed = _build_zoomed_graph(gray, x0, y0, x1, y1, scale, config)
        if zoomed is None:
            continue
        roi_up, roi_skel, roi_bin, roi_adj_z, roi_edges_z, roi_eps_z, roi_juncs_z = zoomed

        vis = cv2.cvtColor(roi_up, cv2.COLOR_GRAY2BGR)
        vis[roi_skel > 0] = [0, 0, 255]
        for ep in roi_eps_z:
            cv2.circle(vis, ep, 3, (0, 255, 0), -1)
        for jn in roi_juncs_z:
            cv2.circle(vis, jn, 4, (255, 0, 255), -1)

        # Draw junction center
        cz = (int((jx - x0) * scale), int((jy - y0) * scale))
        cv2.drawMarker(vis, cz, (0, 255, 255), cv2.MARKER_CROSS, 12, 2)

        path = os.path.join(output_dir,
                            f"debug_zoom_junc_{idx}_{jx}_{jy}.png")
        cv2.imwrite(path, vis)
        logger.debug("Saved zoomed junction debug: %s", path)


# ====================================================================
# F) STATEFUL DIJKSTRA (intensity-aware)
# ====================================================================

def _turn_angle(t_in: Optional[np.ndarray], t_out: Optional[np.ndarray]) -> float:
    if t_in is None or t_out is None:
        return 0.0
    cos_val = max(-1.0, min(1.0, float(np.dot(t_in, t_out))))
    return math.acos(-cos_val)


def _edge_traversal_cost(
    edge: GraphEdge,
    entering_from: Tuple[int, int],
    cost_map: np.ndarray,
    config: JunctionConfig,
) -> float:
    cost = float(edge.length)
    cost += config.mu_curvature * edge.curvature

    if edge.pixels and cost_map is not None:
        h, w = cost_map.shape
        samples = [cost_map[py, px] for px, py in edge.pixels
                   if 0 <= py < h and 0 <= px < w]
        if samples:
            cost += config.nu_darkness * float(np.mean(samples)) * edge.length

    if edge.pixels:
        dx = edge.other_node(entering_from)[0] - entering_from[0]
        if dx < 0:
            cost += config.x_decrease_penalty * abs(dx)

    return cost


def stateful_dijkstra(
    adjacency: Dict[Tuple[int, int], List[GraphEdge]],
    all_edges: List[GraphEdge],
    junctions: Set[Tuple[int, int]],
    junction_pairings: Dict[Tuple[int, int], List[JunctionPairing]],
    start: Tuple[int, int],
    end: Tuple[int, int],
    gray: np.ndarray,
    cost_map: np.ndarray,
    config: JunctionConfig,
    forbidden_edges: Optional[Set[int]] = None,
    curve_intensity: Optional[float] = None,
) -> Optional[Tuple[float, List[int]]]:
    """Direction-aware Dijkstra with intensity-matching cost.

    If curve_intensity is provided, edges whose mean_intensity deviates
    from it are penalised.
    """
    if forbidden_edges is None:
        forbidden_edges = set()

    edge_map = {e.edge_id: e for e in all_edges}
    counter = 0
    pq: List[Tuple[float, int, Tuple[int, int], Optional[int], Tuple[int, ...]]] = []
    counter += 1
    heapq.heappush(pq, (0.0, counter, start, None, ()))
    visited: Dict[Tuple[Tuple[int, int], Optional[int]], float] = {}

    while pq:
        cost, _cnt, node, incoming_eid, path_eids = heapq.heappop(pq)
        state_key = (node, incoming_eid)
        if state_key in visited:
            continue
        visited[state_key] = cost

        if node == end:
            return (cost, list(path_eids))

        for edge in adjacency.get(node, []):
            if edge.edge_id in forbidden_edges or edge.edge_id == incoming_eid:
                continue
            next_node = edge.other_node(node)
            if (next_node, edge.edge_id) in visited:
                continue

            edge_cost = _edge_traversal_cost(edge, node, cost_map, config)

            # Intensity matching
            if curve_intensity is not None:
                intensity_dev = abs(edge.mean_intensity - curve_intensity) / 255.0
                edge_cost += config.alpha_intensity * intensity_dev * edge.length

            # Turn penalty at junctions
            turn_cost = 0.0
            if node in junctions and incoming_eid is not None:
                inc_edge = edge_map.get(incoming_eid)
                if inc_edge is not None:
                    angle = _turn_angle(inc_edge.tangent_at(node), edge.tangent_at(node))
                    turn_cost = config.lambda_turn * (angle ** 2)
                    i_delta = abs(inc_edge.mean_intensity - edge.mean_intensity)
                    turn_cost += config.alpha_intensity * (i_delta / 255.0)
                    for p in junction_pairings.get(node, []):
                        if p.edge_in == incoming_eid and p.edge_out == edge.edge_id:
                            turn_cost *= max(0.1, 1.0 - p.cos_similarity)
                            break

                    logger.debug(
                        "junc (%d,%d) e%d->e%d ang=%.1f° di=%.1f tc=%.1f",
                        node[0], node[1], incoming_eid, edge.edge_id,
                        math.degrees(angle), i_delta, turn_cost,
                    )

            new_cost = cost + edge_cost + turn_cost
            counter += 1
            heapq.heappush(pq, (new_cost, counter, next_node, edge.edge_id,
                                path_eids + (edge.edge_id,)))

    return None


# ====================================================================
# E) BRANCH-AWARE ANCHOR SNAPPING
# ====================================================================

def snap_anchors(
    anchors: List[Tuple[int, int]],
    adjacency: Dict[Tuple[int, int], List[GraphEdge]],
    all_edges: List[GraphEdge],
    gray: np.ndarray,
    config: JunctionConfig,
) -> List[Tuple[Tuple[int, int], float]]:
    """Snap each anchor to the best graph node (direction + intensity aware).

    Returns list of (snapped_node, estimated_intensity).
    """
    all_nodes = list(adjacency.keys())
    if not all_nodes:
        return [(a, 128.0) for a in anchors]

    node_arr = np.array(all_nodes, dtype=np.float64)
    h, w = gray.shape

    # Estimate curve intensity at each raw anchor
    anchor_intensities = []
    for ax, ay in anchors:
        r = 5
        patch = gray[max(0, ay - r):min(h, ay + r + 1),
                      max(0, ax - r):min(w, ax + r + 1)]
        if patch.size > 0:
            sorted_vals = np.sort(patch.ravel())
            q = max(1, len(sorted_vals) // 4)
            anchor_intensities.append(float(np.mean(sorted_vals[:q])))
        else:
            anchor_intensities.append(128.0)

    result: List[Tuple[Tuple[int, int], float]] = []

    for idx, (ax, ay) in enumerate(anchors):
        a_pt = np.array([ax, ay], dtype=np.float64)
        a_intensity = anchor_intensities[idx]

        dists = np.linalg.norm(node_arr - a_pt, axis=1)
        within = dists <= config.snap_radius
        if not within.any():
            result.append((all_nodes[int(np.argmin(dists))], a_intensity))
            continue

        sorted_idx = np.where(within)[0][np.argsort(dists[within])]
        top_k = sorted_idx[:config.snap_k]

        # Expected direction from neighbors
        expected_dir = None
        if len(anchors) > 1:
            if idx == 0:
                expected_dir = np.array([anchors[1][0] - ax, anchors[1][1] - ay], dtype=np.float64)
            elif idx == len(anchors) - 1:
                expected_dir = np.array([ax - anchors[-2][0], ay - anchors[-2][1]], dtype=np.float64)
            else:
                expected_dir = np.array([anchors[idx + 1][0] - anchors[idx - 1][0],
                                         anchors[idx + 1][1] - anchors[idx - 1][1]], dtype=np.float64)
            norm = np.linalg.norm(expected_dir)
            expected_dir = expected_dir / norm if norm > 1e-8 else None

        best_score = float("inf")
        best_node = all_nodes[int(top_k[0])]
        best_int = a_intensity

        for ci in top_k:
            cnode = all_nodes[ci]
            dist_cost = float(dists[ci])
            edges_at = adjacency.get(cnode, [])
            c_int = float(np.mean([e.mean_intensity for e in edges_at])) if edges_at else 128.0
            intensity_cost = abs(c_int - a_intensity)

            dir_cost = 0.0
            if expected_dir is not None and edges_at:
                best_cos = max(
                    (abs(float(np.dot(e.tangent_at(cnode), expected_dir)))
                     for e in edges_at if e.tangent_at(cnode) is not None),
                    default=0.0,
                )
                dir_cost = (1.0 - best_cos) * 20.0

            score = dist_cost + intensity_cost * 0.3 + dir_cost
            if score < best_score:
                best_score = score
                best_node = cnode
                best_int = c_int

        result.append((best_node, best_int))

    return result


# ====================================================================
# G) BEAM SEARCH between anchor pairs
# ====================================================================

def trace_segment_beam(
    start: Tuple[int, int],
    end: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[GraphEdge]],
    all_edges: List[GraphEdge],
    junctions: Set[Tuple[int, int]],
    junction_pairings: Dict[Tuple[int, int], List[JunctionPairing]],
    gray: np.ndarray,
    cost_map: np.ndarray,
    config: JunctionConfig,
    curve_intensity: Optional[float] = None,
    forbidden_edges: Optional[Set[int]] = None,
) -> List[Tuple[float, List[int]]]:
    """Top K shortest paths via progressive edge forbidding."""
    results: List[Tuple[float, List[int]]] = []
    used_sets: List[Set[int]] = []
    cum_forbidden = set(forbidden_edges) if forbidden_edges else set()
    edge_map = {e.edge_id: e for e in all_edges}

    for _ in range(config.beam_width):
        result = stateful_dijkstra(
            adjacency, all_edges, junctions, junction_pairings,
            start, end, gray, cost_map, config,
            forbidden_edges=cum_forbidden if cum_forbidden else None,
            curve_intensity=curve_intensity,
        )
        if result is None:
            break

        cost, eids = result
        eset = set(eids)
        is_dup = any(
            len(eset) > 0 and len(eset & ps) > 0.6 * len(eset)
            for ps in used_sets
        )
        if is_dup:
            cum_forbidden |= eset
            continue

        results.append((cost, eids))
        used_sets.append(eset)
        for eid in eids:
            e = edge_map.get(eid)
            if e and (e.node_a in junctions or e.node_b in junctions):
                cum_forbidden.add(eid)

    results.sort(key=lambda x: x[0])
    return results


# ====================================================================
# Trace full curve through ordered anchors
# ====================================================================

def trace_curve(
    anchors: List[Tuple[int, int]],
    adjacency: Dict[Tuple[int, int], List[GraphEdge]],
    all_edges: List[GraphEdge],
    junctions: Set[Tuple[int, int]],
    junction_pairings: Dict[Tuple[int, int], List[JunctionPairing]],
    gray: np.ndarray,
    cost_map: np.ndarray,
    config: JunctionConfig,
    forbidden_edges: Optional[Set[int]] = None,
) -> Optional[Tuple[List[Tuple[int, int]], float]]:
    """Trace a single curve via branch-aware snapping + beam search.

    Returns (pixel_path, curve_intensity) or None.
    """
    if len(anchors) < 2:
        return None

    snapped = snap_anchors(anchors, adjacency, all_edges, gray, config)
    if not snapped:
        return None

    snapped_nodes = [s[0] for s in snapped]
    estimated_intensity = float(np.median([s[1] for s in snapped]))
    logger.info("trace_curve: %d anchors, intensity=%.0f", len(anchors), estimated_intensity)

    edge_map = {e.edge_id: e for e in all_edges}
    full_edge_ids: List[int] = []

    for i in range(len(snapped_nodes) - 1):
        candidates = trace_segment_beam(
            snapped_nodes[i], snapped_nodes[i + 1],
            adjacency, all_edges, junctions, junction_pairings,
            gray, cost_map, config,
            curve_intensity=estimated_intensity,
            forbidden_edges=forbidden_edges,
        )
        if not candidates:
            logger.warning("trace_curve: no path segment %d->%d", i, i + 1)
            return None
        _, best_eids = candidates[0]
        full_edge_ids.extend(best_eids)

    pixels = _edges_to_pixels(full_edge_ids, edge_map, snapped_nodes[0], snapped_nodes[-1])
    return (pixels, estimated_intensity)


# ====================================================================
# Auto-trace (no anchors)
# ====================================================================

def _extract_curves_by_junction_splitting(
    adjacency: Dict[Tuple[int, int], List[GraphEdge]],
    all_edges: List[GraphEdge],
    endpoints: Set[Tuple[int, int]],
    junctions: Set[Tuple[int, int]],
    junction_pairings: Dict[Tuple[int, int], List[JunctionPairing]],
    config: JunctionConfig,
    image_width: int,
) -> List[List[Tuple[int, int]]]:
    """Union-Find curve extraction using junction pairings."""
    if not all_edges:
        return []

    parent: Dict[int, int] = {}
    rank: Dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent.get(x, x), parent.get(x, x))
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank.get(ra, 0) < rank.get(rb, 0):
            ra, rb = rb, ra
        parent[rb] = ra
        if rank.get(ra, 0) == rank.get(rb, 0):
            rank[ra] = rank.get(ra, 0) + 1

    for e in all_edges:
        parent[e.edge_id] = e.edge_id
        rank[e.edge_id] = 0

    node_to_edges: Dict[Tuple[int, int], List[int]] = {}
    for e in all_edges:
        node_to_edges.setdefault(e.node_a, []).append(e.edge_id)
        node_to_edges.setdefault(e.node_b, []).append(e.edge_id)

    for node, eids in node_to_edges.items():
        if node not in junctions and len(eids) == 2:
            union(eids[0], eids[1])

    for junc, plist in junction_pairings.items():
        seen: Set[Tuple[int, int]] = set()
        for p in plist:
            key = (min(p.edge_in, p.edge_out), max(p.edge_in, p.edge_out))
            if key in seen:
                continue
            seen.add(key)
            union(p.edge_in, p.edge_out)

    components: Dict[int, List[GraphEdge]] = {}
    for e in all_edges:
        components.setdefault(find(e.edge_id), []).append(e)

    min_xspan = image_width * config.min_curve_xspan_ratio
    curves: List[Tuple[int, List[Tuple[int, int]]]] = []

    for comp_edges in components.values():
        all_px: Set[Tuple[int, int]] = set()
        for e in comp_edges:
            for p in e.pixels:
                all_px.add(p)
        if not all_px:
            continue
        xs = [p[0] for p in all_px]
        x_span = max(xs) - min(xs)
        if x_span < min_xspan:
            continue
        curves.append((x_span, sorted(all_px, key=lambda p: (p[0], p[1]))))

    curves.sort(key=lambda c: -c[0])
    logger.info("junction_split: %d components, %d valid curves (x>=%d)",
                len(components), len(curves), int(min_xspan))
    return [px for _, px in curves]


def auto_trace_curves(
    adjacency: Dict[Tuple[int, int], List[GraphEdge]],
    all_edges: List[GraphEdge],
    endpoints: Set[Tuple[int, int]],
    junctions: Set[Tuple[int, int]],
    junction_pairings: Dict[Tuple[int, int], List[JunctionPairing]],
    gray: np.ndarray,
    cost_map: np.ndarray,
    config: JunctionConfig,
    image_width: int,
) -> List[List[Tuple[int, int]]]:
    """Iterative peeling Dijkstra with y-separation."""
    if not endpoints:
        return []

    sorted_eps = sorted(endpoints, key=lambda p: p[0])
    left_eps = [e for e in sorted_eps if e[0] < image_width * 0.30]
    right_eps = [e for e in sorted_eps if e[0] > image_width * 0.70]

    if len(left_eps) < 1 or len(right_eps) < 1:
        n = len(sorted_eps)
        if n >= 2:
            left_eps = sorted_eps[:max(1, n // 3)]
            right_eps = sorted_eps[max(1, 2 * n // 3):]
        else:
            return []

    edge_map = {e.edge_id: e for e in all_edges}
    selected: List[List[Tuple[int, int]]] = []
    global_forbidden: Set[int] = set()
    min_xspan = image_width * config.min_curve_xspan_ratio

    for _ in range(config.max_curves * 3):
        if len(selected) >= config.max_curves:
            break
        best_ratio = float("inf")
        best_px = None
        best_eids = None

        for le in left_eps:
            for re in right_eps:
                if abs(re[0] - le[0]) < min_xspan:
                    continue
                res = stateful_dijkstra(
                    adjacency, all_edges, junctions, junction_pairings,
                    le, re, gray, cost_map, config,
                    forbidden_edges=global_forbidden if global_forbidden else None,
                )
                if res is None:
                    continue
                cost, eids = res
                if not eids:
                    continue
                px = _edges_to_pixels(eids, edge_map, le, re)
                if len(px) < config.min_edge_length:
                    continue
                xs = [p[0] for p in px]
                ratio = cost / max(max(xs) - min(xs) + 1, 1)
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_px = px
                    best_eids = eids

        if best_px is None or best_eids is None:
            break

        mean_y = np.mean([p[1] for p in best_px])
        if any(abs(mean_y - np.mean([p[1] for p in s])) < gray.shape[0] * 0.02
               for s in selected):
            for eid in best_eids:
                e = edge_map.get(eid)
                if e and not (e.node_a in junctions and e.node_b in junctions):
                    global_forbidden.add(eid)
            continue

        selected.append(best_px)
        for eid in best_eids:
            e = edge_map.get(eid)
            if e and not (e.node_a in junctions and e.node_b in junctions):
                global_forbidden.add(eid)

    logger.info("auto_trace: extracted %d curves", len(selected))
    return selected


def _edges_to_pixels(
    edge_ids: List[int],
    edge_map: Dict[int, GraphEdge],
    start: Tuple[int, int],
    end: Tuple[int, int],
) -> List[Tuple[int, int]]:
    if not edge_ids:
        return [start, end] if start != end else [start]
    path: List[Tuple[int, int]] = []
    current = start
    for eid in edge_ids:
        edge = edge_map.get(eid)
        if edge is None:
            continue
        if edge.node_a == current:
            pixels = edge.pixels
        elif edge.node_b == current:
            pixels = list(reversed(edge.pixels))
        else:
            da = abs(edge.node_a[0] - current[0]) + abs(edge.node_a[1] - current[1])
            db = abs(edge.node_b[0] - current[0]) + abs(edge.node_b[1] - current[1])
            pixels = edge.pixels if da <= db else list(reversed(edge.pixels))
        si = 1 if path and pixels and pixels[0] == path[-1] else 0
        path.extend(pixels[si:])
        current = pixels[-1] if pixels else current
    return path


# ====================================================================
# H) GLOBAL CONSISTENCY CHECK
# ====================================================================

def check_crossings(
    curves_pixel: Dict[str, List[Tuple[float, float]]],
) -> List[Tuple[str, str, float]]:
    """Detect crossings between curves at sampled x positions."""
    labels = sorted(curves_pixel.keys())
    if len(labels) < 2:
        return []

    x_lo = max(min(p[0] for p in curves_pixel[l]) for l in labels)
    x_hi = min(max(p[0] for p in curves_pixel[l]) for l in labels)
    if x_lo >= x_hi:
        return []

    x_samples = np.linspace(x_lo, x_hi, 50)

    def interp_y(pts, xs):
        arr = np.array(pts)
        order = np.argsort(arr[:, 0])
        return np.interp(xs, arr[order, 0], arr[order, 1])

    y_arr = {l: interp_y(curves_pixel[l], x_samples) for l in labels}
    crossings = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            diff = y_arr[labels[i]] - y_arr[labels[j]]
            sc = np.where(np.diff(np.sign(diff)) != 0)[0]
            for s in sc:
                crossings.append((labels[i], labels[j], float(x_samples[s])))

    if crossings:
        logger.warning("check_crossings: %d crossings", len(crossings))
    return crossings


# ====================================================================
# I) POSTPROCESS
# ====================================================================

def postprocess_curve(
    pixels: List[Tuple[int, int]],
    config: JunctionConfig,
) -> List[Tuple[float, float]]:
    if len(pixels) < 4:
        return [(float(p[0]), float(p[1])) for p in pixels]

    arr = np.array(pixels, dtype=np.float64)
    arr = arr[np.argsort(arr[:, 0])]

    unique_x = np.unique(arr[:, 0])
    x_c, y_c = [], []
    for ux in unique_x:
        x_c.append(ux)
        y_c.append(np.median(arr[arr[:, 0] == ux, 1]))
    x_c, y_c = np.array(x_c), np.array(y_c)

    if len(x_c) < 4:
        return list(zip(x_c.tolist(), y_c.tolist()))

    try:
        coeffs = np.polyfit(x_c, y_c, min(3, len(x_c) - 1))
        residuals = np.abs(y_c - np.polyval(coeffs, x_c))
        mad = np.median(residuals)
        if mad > 0:
            inlier = residuals < 4.0 * mad
            if np.sum(inlier) >= 4:
                x_c, y_c = x_c[inlier], y_c[inlier]
    except Exception:
        pass

    try:
        s = config.spline_smoothing if config.spline_smoothing > 0 else len(x_c) * 0.5
        k = min(3, len(x_c) - 1)
        spline = UnivariateSpline(x_c, y_c, k=k, s=s)
        xr = np.linspace(x_c[0], x_c[-1], config.resample_n)
        return list(zip(xr.tolist(), spline(xr).tolist()))
    except Exception:
        return list(zip(x_c.tolist(), y_c.tolist()))


# ====================================================================
# J) PIXEL -> DATA MAPPING
# ====================================================================

def deskew_image(image: np.ndarray) -> Tuple[np.ndarray, float]:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                             minLineLength=100, maxLineGap=10)
    if lines is None:
        return image, 0.0
    h_angles = [math.degrees(math.atan2(l[0][3] - l[0][1], l[0][2] - l[0][0]))
                for l in lines
                if abs(math.degrees(math.atan2(l[0][3] - l[0][1], l[0][2] - l[0][0]))) < 10]
    if not h_angles:
        return image, 0.0
    skew = float(np.median(h_angles))
    if abs(skew) < 0.1:
        return image, 0.0
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), skew, 1.0)
    deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
    logger.info("deskew_image: rotated %.2f degrees", skew)
    return deskewed, skew


def calibrate_from_3points(calib_points: List[Dict[str, Any]]) -> AffineMapping:
    if len(calib_points) < 3:
        raise ValueError("Need >= 3 calibration points")
    src = np.array([p["pixel"] for p in calib_points], dtype=np.float64)
    dst = np.array([p["data"] for p in calib_points], dtype=np.float64)
    n = len(src)
    A = np.zeros((2 * n, 6))
    bv = np.zeros(2 * n)
    for i in range(n):
        A[2 * i] = [src[i, 0], src[i, 1], 1, 0, 0, 0]
        A[2 * i + 1] = [0, 0, 0, src[i, 0], src[i, 1], 1]
        bv[2 * i] = dst[i, 0]
        bv[2 * i + 1] = dst[i, 1]
    params, _, _, _ = np.linalg.lstsq(A, bv, rcond=None)
    a, b, tx, c, d, ty = params
    p2d = np.array([[a, b, tx], [c, d, ty]])
    M2 = np.array([[a, b], [c, d]])
    try:
        Mi = np.linalg.inv(M2)
    except np.linalg.LinAlgError:
        Mi = np.eye(2)
    ti = -Mi @ np.array([tx, ty])
    d2p = np.array([[Mi[0, 0], Mi[0, 1], ti[0]], [Mi[1, 0], Mi[1, 1], ti[1]]])
    errors = [math.hypot(*(d2p @ np.append(p2d @ np.append(src[i], 1), 1) - src[i]))
              for i in range(n)]
    return AffineMapping(matrix_p2d=p2d, matrix_d2p=d2p,
                          error_px=float(np.mean(errors)))


def calibrate_from_2points_per_axis(
    x_refs: List[Dict[str, float]],
    y_refs: List[Dict[str, float]],
    plot_area: Tuple[int, int, int, int],
) -> AffineMapping:
    if len(x_refs) < 2 or len(y_refs) < 2:
        raise ValueError("Need >= 2 refs per axis")
    sx, tx = np.polyfit([r["pixel"] for r in x_refs], [r["value"] for r in x_refs], 1)
    sy, ty = np.polyfit([r["pixel"] for r in y_refs], [r["value"] for r in y_refs], 1)
    p2d = np.array([[sx, 0, tx], [0, sy, ty]])
    isx = 1 / sx if abs(sx) > 1e-12 else 0
    isy = 1 / sy if abs(sy) > 1e-12 else 0
    d2p = np.array([[isx, 0, -tx * isx], [0, isy, -ty * isy]])
    return AffineMapping(matrix_p2d=p2d, matrix_d2p=d2p, error_px=0.0)


def map_pixels_to_data(
    pixels: List[Tuple[float, float]], mapping: AffineMapping,
) -> List[Tuple[float, float]]:
    return [(float((mapping.matrix_p2d @ np.array([px, py, 1.0]))[0]),
             float((mapping.matrix_p2d @ np.array([px, py, 1.0]))[1]))
            for px, py in pixels]


# ====================================================================
# K) DEBUG OUTPUTS
# ====================================================================

def save_debug_skeleton(image: np.ndarray, skeleton: np.ndarray, path: str) -> None:
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    base[skeleton > 0] = [0, 0, 255]
    cv2.imwrite(path, base)


def save_debug_graph(
    image: np.ndarray,
    all_edges: List[GraphEdge],
    endpoints: Set[Tuple[int, int]],
    junctions: Set[Tuple[int, int]],
    path: str,
) -> None:
    debug = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    for edge in all_edges:
        ni = edge.mean_intensity / 255.0
        color = (int(255 * ni), int(255 * ni), int(255 * (1 - ni)))
        for p in edge.pixels:
            if 0 <= p[1] < debug.shape[0] and 0 <= p[0] < debug.shape[1]:
                debug[p[1], p[0]] = color
    for ep in endpoints:
        cv2.circle(debug, ep, 4, (0, 255, 0), -1)
    for jn in junctions:
        cv2.circle(debug, jn, 5, (0, 0, 255), -1)
    cv2.imwrite(path, debug)


def save_debug_anchors(
    image: np.ndarray,
    raw_anchors: Dict[str, List[Tuple[int, int]]],
    snapped_anchors: Dict[str, List[Tuple[Tuple[int, int], float]]],
    path: str,
) -> None:
    debug = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    colors = [(0, 0, 255), (255, 0, 0), (0, 180, 0), (0, 165, 255),
              (180, 0, 180), (0, 255, 255), (128, 0, 0), (0, 128, 128)]
    for idx, (label, raw_pts) in enumerate(raw_anchors.items()):
        col = colors[idx % len(colors)]
        snapped = snapped_anchors.get(label, [])
        for i, (ax, ay) in enumerate(raw_pts):
            cv2.circle(debug, (ax, ay), 6, col, 1)
            cv2.putText(debug, str(i), (ax + 8, ay - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
            if i < len(snapped):
                sx, sy = snapped[i][0]
                cv2.circle(debug, (sx, sy), 4, col, -1)
                cv2.line(debug, (ax, ay), (sx, sy), col, 1)
                if i < len(snapped) - 1:
                    nx, ny = snapped[i + 1][0]
                    d = math.hypot(nx - sx, ny - sy)
                    if d > 0:
                        cv2.arrowedLine(debug, (sx, sy),
                                        (int(sx + (nx - sx) / d * 20),
                                         int(sy + (ny - sy) / d * 20)),
                                        col, 1, tipLength=0.3)
    cv2.imwrite(path, debug)


def save_debug_curve_paths(
    image: np.ndarray, label: str, pixels: List[Tuple[int, int]], path: str,
) -> None:
    debug = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    if pixels:
        pts = np.array([(int(p[0]), int(p[1])) for p in pixels], dtype=np.int32)
        cv2.polylines(debug, [pts], False, (0, 0, 255), 2)
    cv2.putText(debug, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(path, debug)


def generate_qa_overlay(
    image: np.ndarray,
    curves: Dict[str, List[Tuple[float, float]]],
    output_path: str,
) -> np.ndarray:
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    colors = [(0, 0, 255), (255, 0, 0), (0, 180, 0), (0, 165, 255),
              (180, 0, 180), (0, 255, 255), (128, 0, 0), (0, 128, 128),
              (255, 128, 0), (0, 255, 128), (128, 0, 255), (255, 0, 128)]
    for idx, (label, points) in enumerate(curves.items()):
        col = colors[idx % len(colors)]
        pts = np.array([(int(p[0]), int(p[1])) for p in points], dtype=np.int32)
        if len(pts) > 1:
            cv2.polylines(overlay, [pts], False, col, 2)
        if len(pts) > 0:
            mid = pts[len(pts) // 2]
            cv2.putText(overlay, label, (mid[0], mid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    cv2.imwrite(output_path, overlay)
    logger.info("Saved overlay to %s", output_path)
    return overlay


def generate_debug_view(
    image: np.ndarray,
    adjacency: Dict[Tuple[int, int], List[GraphEdge]],
    all_edges: List[GraphEdge],
    endpoints: Set[Tuple[int, int]],
    junctions: Set[Tuple[int, int]],
    junction_pairings: Dict[Tuple[int, int], List[JunctionPairing]],
    curves: Dict[str, List[Tuple[float, float]]],
    output_path: str,
) -> np.ndarray:
    debug = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    for edge in all_edges:
        for p in edge.pixels:
            if 0 <= p[1] < debug.shape[0] and 0 <= p[0] < debug.shape[1]:
                debug[p[1], p[0]] = (180, 180, 180)
    for ep in endpoints:
        cv2.circle(debug, ep, 4, (0, 255, 0), -1)
    for jn in junctions:
        cv2.circle(debug, jn, 5, (0, 0, 255), -1)

    edge_map = {e.edge_id: e for e in all_edges}
    for jn, plist in junction_pairings.items():
        for p in plist:
            e_in = edge_map.get(p.edge_in)
            e_out = edge_map.get(p.edge_out)
            if not e_in or not e_out:
                continue
            t_in = e_in.tangent_at(jn)
            t_out = e_out.tangent_at(jn)
            if t_in is not None:
                cv2.arrowedLine(debug,
                                (int(jn[0] - t_in[0] * 15), int(jn[1] - t_in[1] * 15)),
                                jn, (255, 200, 0), 1, tipLength=0.3)
            if t_out is not None:
                cv2.arrowedLine(debug, jn,
                                (int(jn[0] + t_out[0] * 15), int(jn[1] + t_out[1] * 15)),
                                (0, 200, 255), 1, tipLength=0.3)

    curve_colors = [(0, 0, 255), (255, 0, 0), (0, 180, 0),
                    (0, 165, 255), (180, 0, 180), (0, 255, 255)]
    for idx, (label, points) in enumerate(curves.items()):
        col = curve_colors[idx % len(curve_colors)]
        pts = np.array([(int(p[0]), int(p[1])) for p in points], dtype=np.int32)
        if len(pts) > 1:
            cv2.polylines(debug, [pts], False, col, 2)

    cv2.putText(debug, "Green=ep  Red=junc  Arrows=pairings",
                (10, debug.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(output_path, debug)
    return debug


# ====================================================================
# Export
# ====================================================================

def export_curves_json(curves: Dict[str, List[Tuple[float, float]]], path: str) -> None:
    data = {l: [{"x": p[0], "y": p[1]} for p in pts] for l, pts in curves.items()}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Exported %d curves to %s", len(curves), path)


def export_curves_csv(curves: Dict[str, List[Tuple[float, float]]], path: str) -> None:
    labels = sorted(curves.keys())
    max_len = max((len(curves[l]) for l in labels), default=0)
    lines = [",".join(f"{l}_x,{l}_y" for l in labels)]
    for i in range(max_len):
        row = []
        for l in labels:
            pts = curves[l]
            row.append(f"{pts[i][0]:.6f},{pts[i][1]:.6f}" if i < len(pts) else ",")
        lines.append(",".join(row))
    with open(path, "w") as f:
        f.write("\n".join(lines))
    logger.info("Exported %d curves to %s", len(curves), path)


# ====================================================================
# Summary
# ====================================================================

def summarize_image(
    image: np.ndarray,
    skeleton: np.ndarray,
    endpoints: Set[Tuple[int, int]],
    junctions: Set[Tuple[int, int]],
    all_edges: List[GraphEdge],
    num_curves: int,
) -> str:
    h, w = image.shape[:2]
    est = max(1, len(endpoints) // 2)
    total_px = sum(e.length for e in all_edges)
    pct = 100.0 * total_px / max(h * w, 1)
    sev = "heavy" if len(junctions) > 5 else "moderate" if len(junctions) > 2 else "minor"
    return (f"B/W plot ({w}x{h}px), ~{est} curves. "
            f"{len(junctions)} junctions ({sev}). "
            f"Skeleton: {len(all_edges)} edges, {len(endpoints)} eps, {pct:.1f}% coverage. "
            f"Extracted {num_curves} curve(s).")


# ====================================================================
# MAIN PIPELINE
# ====================================================================

def digitize(
    image_path: str,
    calib_path: Optional[str] = None,
    anchors_path: Optional[str] = None,
    output_dir: str = "output_junction",
    config: Optional[JunctionConfig] = None,
    num_curves: int = 0,
) -> Dict[str, Any]:
    """Main entry point."""
    if config is None:
        config = JunctionConfig()

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    os.makedirs(output_dir, exist_ok=True)
    logger.info("digitize: loaded %s (%dx%d)", image_path, image.shape[1], image.shape[0])

    deskewed, skew_angle = deskew_image(image)
    skeleton, binary, gray, cost_map = preprocess(deskewed, config)

    debug_dir = config.debug_dir or output_dir
    if config.debug:
        cv2.imwrite(os.path.join(debug_dir, "01_skeleton.png"), skeleton)
        cv2.imwrite(os.path.join(debug_dir, "02_binary.png"), binary)
        cost_vis = (cost_map / (cost_map.max() + 1e-8) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_dir, "03_cost_map.png"),
                     cv2.applyColorMap(cost_vis, cv2.COLORMAP_JET))

    adjacency, all_edges, endpoints, junctions = build_graph(skeleton, gray, cost_map, config)

    if not all_edges:
        return {"curves": {}, "summary": "No curves detected.", "files_written": []}

    junction_pairings = disambiguate_junctions(adjacency, junctions, config)

    # ---- MULTI-SCALE ZOOM JUNCTION REFINEMENT (pre-tracing) ----
    num_refined = 0
    if config.refine_enabled and junctions:
        junction_pairings, num_refined = refine_junctions_multiscale(
            adjacency, all_edges, junctions, junction_pairings, gray, config)
        logger.info("digitize: multi-scale refined %d junctions", num_refined)

    if config.debug:
        save_debug_skeleton(deskewed, skeleton, os.path.join(debug_dir, "debug_skeleton.png"))
        save_debug_graph(deskewed, all_edges, endpoints, junctions,
                         os.path.join(debug_dir, "debug_graph.png"))
        if config.refine_enabled and num_refined > 0:
            save_debug_junction_zoom(
                deskewed, junctions, adjacency, config, debug_dir)

    # Calibration
    mapping: Optional[AffineMapping] = None
    if calib_path and os.path.isfile(calib_path):
        with open(calib_path) as f:
            cd = json.load(f)
        if cd.get("mode") == "3point":
            mapping = calibrate_from_3points(cd["points"])
        elif cd.get("mode") == "2axis":
            mapping = calibrate_from_2points_per_axis(cd["x_refs"], cd["y_refs"],
                                                       tuple(cd["plot_area"]))

    curves_pixel: Dict[str, List[Tuple[float, float]]] = {}
    files_written: List[str] = []

    if anchors_path and os.path.isfile(anchors_path):
        # ---- ANCHORED MODE (direct pixel tracing, one curve at a time) ----
        with open(anchors_path) as f:
            ad = json.load(f)

        for cs in ad.get("curves", []):
            label = cs.get("label", f"curve_{len(curves_pixel)}")
            pts = [(int(a[0]), int(a[1])) for a in cs["anchors"]]

            pixels = trace_single_curve_anchored(
                pts, skeleton, gray, config, junctions=junctions)

            if pixels is not None and len(pixels) >= 4:
                # Junction zoom refinement along the anchored polyline
                if config.refine_enabled and junctions:
                    pixels = refine_curve_pixels_at_junctions(
                        pixels, junctions, gray, config)
                curves_pixel[label] = postprocess_curve(pixels, config)
                if config.debug:
                    save_debug_curve_paths(deskewed, label, pixels,
                                           os.path.join(debug_dir, f"debug_paths_{label}.png"))
            else:
                logger.warning("digitize: failed to trace '%s'", label)
    else:
        # ---- AUTO MODE ----
        raw_curves = _extract_curves_by_junction_splitting(
            adjacency, all_edges, endpoints, junctions,
            junction_pairings, config, image_width=skeleton.shape[1])

        if len(raw_curves) < (num_curves if num_curves > 0 else 2):
            logger.info("junction-split: %d, supplementing", len(raw_curves))
            extra = auto_trace_curves(
                adjacency, all_edges, endpoints, junctions,
                junction_pairings, gray, cost_map, config,
                image_width=skeleton.shape[1])
            for dc in extra:
                my = np.mean([p[1] for p in dc])
                if not any(abs(my - np.mean([p[1] for p in rc])) < gray.shape[0] * 0.02
                           for rc in raw_curves):
                    raw_curves.append(dc)

        raw_curves.sort(key=lambda c: np.mean([p[1] for p in c]))
        expected = num_curves if num_curves > 0 else len(raw_curves)
        for i, px in enumerate(raw_curves[:expected]):
            # Post-tracing junction pixel refinement
            if config.refine_enabled and junctions:
                px = refine_curve_pixels_at_junctions(
                    px, junctions, gray, config)
            label = f"curve_{i}"
            curves_pixel[label] = postprocess_curve(px, config)
            if config.debug:
                save_debug_curve_paths(deskewed, label, px,
                                       os.path.join(debug_dir, f"debug_paths_{label}.png"))

    # Map to data coordinates
    curves_data = {}
    if mapping:
        for l, pts in curves_pixel.items():
            curves_data[l] = map_pixels_to_data(pts, mapping)
    else:
        curves_data = curves_pixel

    # Export
    jp = os.path.join(output_dir, "curves.json")
    export_curves_json(curves_data, jp)
    files_written.append(jp)

    cp = os.path.join(output_dir, "curves.csv")
    export_curves_csv(curves_data, cp)
    files_written.append(cp)

    # Per-curve CSVs
    for label, pts in curves_data.items():
        csv_path = os.path.join(output_dir, f"{label}.csv")
        with open(csv_path, "w") as fh:
            fh.write("x,y\n")
            for x, y in pts:
                fh.write(f"{x:.6f},{y:.6f}\n")
        files_written.append(csv_path)

    op = os.path.join(output_dir, "overlay.png")
    generate_qa_overlay(deskewed, curves_pixel, op)
    files_written.append(op)

    dp = os.path.join(output_dir, "debug_junctions.png")
    generate_debug_view(deskewed, adjacency, all_edges, endpoints, junctions,
                        junction_pairings, curves_pixel, dp)
    files_written.append(dp)

    summary = summarize_image(deskewed, skeleton, endpoints, junctions,
                               all_edges, len(curves_pixel))

    # Compute reprojection error: mean pixel distance from curve mask to traced curve
    reproj_errors = []
    skel_px = set(zip(*np.where(skeleton > 0)[::-1]))  # (x, y)
    for label, pts in curves_pixel.items():
        for px, py in pts:
            ix, iy = int(round(px)), int(round(py))
            min_d = min((math.hypot(ix - sx, iy - sy)
                         for sx, sy in skel_px if abs(ix - sx) < 5 and abs(iy - sy) < 5),
                        default=0.0)
            reproj_errors.append(min_d)
    avg_reproj = float(np.mean(reproj_errors)) if reproj_errors else 0.0
    logger.info("Avg reprojection error: %.2f px", avg_reproj)

    return {
        "curves": curves_data,
        "curves_pixel": curves_pixel,
        "summary": summary,
        "skew_angle": skew_angle,
        "num_junctions": len(junctions),
        "num_endpoints": len(endpoints),
        "num_edges": len(all_edges),
        "junction_pairings_count": sum(len(v) for v in junction_pairings.values()),
        "num_junctions_refined": num_refined,
        "avg_reprojection_error_px": avg_reproj,
        "files_written": files_written,
        "mapping_error_px": mapping.error_px if mapping else None,
    }
