"""
Ordered DP multi-curve tracker for BW extraction.

Extracts *N* non-crossing curves from a cleaned binary / skeleton image
using sequential dynamic-programming with exclusion-band masking.

Algorithm
---------
1. Build a **likelihood map** *L(x, y)* from the distance-transform of
   the binary mask (centre of thick strokes scores highest).
2. For each x-column, cluster skeleton/binary y-positions into peaks
   (merge nearby y's, take median per cluster) → candidate y-positions.
3. Extract curves **one at a time, ordered by mean y** (top → bottom in
   image coords).  Each curve is the minimum-cost path through the
   candidate grid using dynamic programming:

       cost(x, y) = −L(x, y) + λ_jump · |y − y_prev|
                   + λ_curv · |(y − y_prev) − (y_prev − y_prev2)|

   subject to |y − y_prev| ≤ max_jump.
4. After extracting a curve, **dilate a band** around its path and zero
   out *L* in that band so the next extraction cannot steal its pixels.
5. Sort final curves by their median y at the image mid-x column.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

logger = logging.getLogger(__name__)


# ====================================================================
# 1.  Likelihood map
# ====================================================================

def build_likelihood_map(binary: np.ndarray) -> np.ndarray:
    """Return a float likelihood map.  High on stroke centres, zero outside.

    Uses the distance transform of the binary mask so that the centre of
    thick strokes gets a higher score than the edges.
    """
    dt = distance_transform_edt(binary)
    # Normalise to [0, 1]
    mx = dt.max()
    if mx > 0:
        dt = dt / mx
    return dt.astype(np.float32)


# ====================================================================
# 2.  Column candidate extraction
# ====================================================================

def _column_candidates(
    binary: np.ndarray,
    *,
    min_cluster_gap: int = 5,
) -> Dict[int, List[int]]:
    """For every x-column, return a sorted list of candidate y-positions.

    Close y-values (within *min_cluster_gap*) are merged by taking the
    group median, avoiding duplicate paths along stroke thickness.
    """
    h, w = binary.shape
    candidates: Dict[int, List[int]] = {}

    for x in range(w):
        ys = np.where(binary[:, x])[0]
        if len(ys) == 0:
            continue

        # Cluster consecutive/close y values
        groups: List[List[int]] = []
        cur_group: List[int] = [int(ys[0])]
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] <= min_cluster_gap:
                cur_group.append(int(ys[i]))
            else:
                groups.append(cur_group)
                cur_group = [int(ys[i])]
        groups.append(cur_group)

        meds = sorted(int(np.median(g)) for g in groups)
        candidates[x] = meds

    return candidates


# ====================================================================
# 3.  Single-curve DP extraction
# ====================================================================

def _extract_one_curve_dp(
    likelihood: np.ndarray,
    candidates: Dict[int, List[int]],
    *,
    max_jump: int = 25,
    jump_weight: float = 0.3,
    curvature_weight: float = 0.1,
    min_span_ratio: float = 0.30,
    y_bias: Optional[float] = None,
    y_bias_weight: float = 0.002,
    y_min_bound: Optional[Dict[int, float]] = None,
    y_max_bound: Optional[Dict[int, float]] = None,
) -> Optional[List[Tuple[int, int]]]:
    """Extract a single curve using column-wise DP.

    Parameters
    ----------
    likelihood : (H, W) float32
        Likelihood map (higher = more likely on curve).
    candidates : dict x → sorted list of y
        Candidate y-positions per column.
    max_jump : int
        Maximum allowed |y − y_prev| between consecutive columns.
    jump_weight : float
        Weight for the jump penalty term.
    curvature_weight : float
        Weight for the second-derivative penalty term.
    min_span_ratio : float
        Minimum fraction of x-columns that must be covered.
    y_bias : float or None
        If given, add a mild penalty pulling the path toward this y
        (helps extract ordered curves from top to bottom).
    y_bias_weight : float
        Weight for the y_bias penalty.

    Returns
    -------
    List of (x, y) or None if no valid path found.
    """
    h, w = likelihood.shape
    x_cols = sorted(candidates.keys())
    if len(x_cols) < 10:
        return None

    # Check feasibility: must span enough of the image
    x_min, x_max = x_cols[0], x_cols[-1]
    if (x_max - x_min) < w * min_span_ratio:
        return None

    # DP tables — store cost and back-pointers
    # cost[x] = {y: min_cost}
    INF = float("inf")

    # Initialise first column
    x0 = x_cols[0]
    cost_prev: Dict[int, float] = {}
    back_prev: Dict[int, Tuple[int, int]] = {}   # y -> (prev_x, prev_y) — unused for first col
    slope_prev: Dict[int, float] = {}             # y -> estimated dy/dx

    for y in candidates.get(x0, []):
        if y_min_bound is not None and x0 in y_min_bound and y < y_min_bound[x0]:
            continue
        if y_max_bound is not None and x0 in y_max_bound and y > y_max_bound[x0]:
            continue
        lk = float(likelihood[y, x0])
        c = -lk
        if y_bias is not None:
            c += y_bias_weight * abs(y - y_bias)
        cost_prev[y] = c
        slope_prev[y] = 0.0

    # Forward pass — column by column
    back: Dict[int, Dict[int, Tuple[int, int]]] = {}  # x -> {y -> (prev_x, prev_y)}

    prev_x = x0
    for xi in range(1, len(x_cols)):
        x = x_cols[xi]
        x_gap = x - prev_x
        if x_gap > max_jump * 3:
            # Too large a gap — reset DP from this column
            cost_prev = {}
            slope_prev = {}
            for y in candidates.get(x, []):
                if y_min_bound is not None and x in y_min_bound and y < y_min_bound[x]:
                    continue
                if y_max_bound is not None and x in y_max_bound and y > y_max_bound[x]:
                    continue
                lk = float(likelihood[y, x])
                c = -lk
                if y_bias is not None:
                    c += y_bias_weight * abs(y - y_bias)
                cost_prev[y] = c
                slope_prev[y] = 0.0
            back[x] = {}
            prev_x = x
            continue

        cost_cur: Dict[int, float] = {}
        back_cur: Dict[int, Tuple[int, int]] = {}
        slope_cur: Dict[int, float] = {}
        cands_y = candidates.get(x, [])

        for y in cands_y:
            if y_min_bound is not None and x in y_min_bound and y < y_min_bound[x]:
                continue
            if y_max_bound is not None and x in y_max_bound and y > y_max_bound[x]:
                continue
            lk = float(likelihood[y, x])
            best_cost = INF
            best_py = -1

            for py, pc in cost_prev.items():
                dy = abs(y - py)
                if dy > max_jump * max(x_gap, 1):
                    continue

                jump_cost = jump_weight * dy

                # Curvature: deviation from predicted position
                predicted_y = py + slope_prev.get(py, 0.0) * x_gap
                curv_cost = curvature_weight * abs(y - predicted_y)

                total = pc + (-lk) + jump_cost + curv_cost
                if y_bias is not None:
                    total += y_bias_weight * abs(y - y_bias)

                if total < best_cost:
                    best_cost = total
                    best_py = py

            if best_py >= 0:
                cost_cur[y] = best_cost
                back_cur[y] = (prev_x, best_py)
                dy = y - best_py
                old_slope = slope_prev.get(best_py, 0.0)
                # Exponential moving average of slope
                slope_cur[y] = 0.7 * old_slope + 0.3 * (dy / max(x_gap, 1))
            else:
                # No predecessor — start fresh
                cost_cur[y] = -lk
                back_cur[y] = (-1, -1)
                slope_cur[y] = 0.0

        back[x] = back_cur
        cost_prev = cost_cur
        slope_prev = slope_cur
        prev_x = x

    # Back-trace from the lowest-cost endpoint
    if not cost_prev:
        return None

    best_y = min(cost_prev, key=cost_prev.get)
    path: List[Tuple[int, int]] = [(prev_x, best_y)]

    cx, cy = prev_x, best_y
    while cx in back and cy in back[cx]:
        px, py = back[cx][cy]
        if px < 0:
            break
        path.append((px, py))
        cx, cy = px, py

    path.reverse()

    # Check span coverage
    if len(path) < 10:
        return None
    path_x_span = path[-1][0] - path[0][0]
    if path_x_span < w * min_span_ratio:
        return None

    return path


def _path_to_y_map(path: List[Tuple[int, int]], width: int) -> Dict[int, float]:
    """Interpolate path y-values for each x across image width."""
    if not path:
        return {}

    arr = np.array(path, dtype=np.float64)
    xs = arr[:, 0]
    ys = arr[:, 1]

    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    ux, inv = np.unique(xs.astype(int), return_inverse=True)
    uy = np.array([np.median(ys[inv == i]) for i in range(len(ux))], dtype=np.float64)

    if len(ux) == 1:
        return {int(x): float(uy[0]) for x in range(width)}

    x_full = np.arange(width, dtype=np.float64)
    y_full = np.interp(x_full, ux.astype(np.float64), uy)
    return {int(x): float(y) for x, y in zip(x_full, y_full)}


# ====================================================================
# 4.  Exclusion-band masking
# ====================================================================

def _mask_exclusion_band(
    likelihood: np.ndarray,
    candidates: Dict[int, List[int]],
    path: List[Tuple[int, int]],
    *,
    band_radius: int = 8,
) -> None:
    """Zero-out *likelihood* and remove *candidates* near *path* (in-place)."""
    h, w = likelihood.shape

    for px, py in path:
        y_lo = max(0, py - band_radius)
        y_hi = min(h, py + band_radius + 1)
        x_lo = max(0, px - 1)
        x_hi = min(w, px + 2)
        likelihood[y_lo:y_hi, x_lo:x_hi] = 0

    # Rebuild candidates for affected columns
    for px, py in path:
        if px in candidates:
            candidates[px] = [
                y for y in candidates[px]
                if abs(y - py) > band_radius
            ]
            if not candidates[px]:
                del candidates[px]


# ====================================================================
# 5.  Estimate stroke width (for exclusion band sizing)
# ====================================================================

def _estimate_stroke_width(binary: np.ndarray) -> int:
    """Estimate median stroke width from distance-transform percentiles."""
    dt = distance_transform_edt(binary)
    on_pixels = dt[binary]
    if len(on_pixels) == 0:
        return 5
    # The distance-transform value at the centre of a stroke ≈ half-width
    p75 = float(np.percentile(on_pixels, 75))
    return max(3, int(p75 * 2 + 2))


# ====================================================================
# 6.  Public entry point
# ====================================================================

def extract_curves_dp(
    binary: np.ndarray,
    skeleton: np.ndarray,
    num_curves: int,
    *,
    max_jump: int = 0,
    jump_weight: float = 0.3,
    curvature_weight: float = 0.1,
    min_span_ratio: float = 0.30,
    cluster_gap: int = 0,
) -> Tuple[Dict[int, List[Tuple[int, int]]], Dict[str, Any]]:
    """Extract *num_curves* non-crossing curves via ordered DP tracking.

    Parameters
    ----------
    binary : (H, W) bool
        Cleaned binary image (grid/text already removed).
    skeleton : (H, W) bool
        Skeletonised binary image.
    num_curves : int
        Number of curves to extract.
    max_jump, jump_weight, curvature_weight, min_span_ratio
        DP tuning (0 = auto for max_jump).
    cluster_gap : int
        Min gap between y clusters per column (0 = auto).

    Returns
    -------
    curves : Dict[int, List[(x, y)]]
        Curve index → ordered pixel coordinates (plot-area local).
    debug : dict
        Debug info: exclusion_bands, stroke_width, etc.
    """
    h, w = binary.shape

    # Auto-compute parameters
    if max_jump <= 0:
        max_jump = max(10, int(h * 0.06))
    if cluster_gap <= 0:
        cluster_gap = max(3, int(h * 0.008))

    stroke_w = _estimate_stroke_width(binary)
    band_radius = max(stroke_w + 2, 6)

    logger.info(
        "extract_curves_dp: h=%d w=%d target=%d max_jump=%d stroke_w=%d band_r=%d",
        h, w, num_curves, max_jump, stroke_w, band_radius,
    )

    # 1. Build likelihood map from binary (NOT skeleton — skeleton is too thin
    #    for distance-transform weighting, but we use binary which has stroke width)
    likelihood = build_likelihood_map(binary)

    # 2. Build column candidates from skeleton (thinner → cleaner peaks)
    use_for_candidates = skeleton if skeleton.any() else binary
    candidates = _column_candidates(use_for_candidates, min_cluster_gap=cluster_gap)

    if not candidates:
        logger.warning("extract_curves_dp: no candidates found")
        return {}, {"stroke_width": stroke_w, "exclusion_bands": []}

    # 3. Sequential extraction: top-to-bottom by y_bias
    #    First pass: estimate the y-range of the data
    all_ys: List[int] = []
    for ys in candidates.values():
        all_ys.extend(ys)
    if not all_ys:
        return {}, {"stroke_width": stroke_w, "exclusion_bands": []}

    y_min_data = min(all_ys)
    y_max_data = max(all_ys)
    y_range = max(y_max_data - y_min_data, 1)

    curves: Dict[int, List[Tuple[int, int]]] = {}
    exclusion_bands: List[List[Tuple[int, int]]] = []
    lower_bound_by_x: Optional[Dict[int, float]] = None

    # Generate y_bias targets spaced evenly across data range
    if num_curves <= 1:
        y_targets = [float(y_min_data + y_range / 2)]
    else:
        y_targets = [
            float(y_min_data + y_range * i / (num_curves - 1))
            for i in range(num_curves)
        ]

    # Sort targets top-to-bottom (lowest y first in image coords = topmost)
    y_targets.sort()

    attempts = 0
    max_attempts = num_curves * 3
    target_idx = 0

    while len(curves) < num_curves and attempts < max_attempts and target_idx < len(y_targets):
        attempts += 1
        y_bias = y_targets[target_idx]

        path = _extract_one_curve_dp(
            likelihood,
            candidates,
            max_jump=max_jump,
            jump_weight=jump_weight,
            curvature_weight=curvature_weight,
            min_span_ratio=min_span_ratio,
            y_bias=y_bias,
            y_bias_weight=0.003,
            y_min_bound=lower_bound_by_x,
        )

        if path is None:
            target_idx += 1
            continue

        # Store and mask
        idx = len(curves)
        curves[idx] = path
        exclusion_bands.append(path)

        # Hard ordering for subsequent curves: next curves must stay below
        # this curve by at least a margin (image coords: larger y = lower).
        path_y_map = _path_to_y_map(path, w)
        this_lower = {
            x: y + max(2, int(band_radius * 0.6))
            for x, y in path_y_map.items()
        }
        if lower_bound_by_x is None:
            lower_bound_by_x = this_lower
        else:
            for x, y in this_lower.items():
                prev = lower_bound_by_x.get(x, -1e9)
                if y > prev:
                    lower_bound_by_x[x] = y

        _mask_exclusion_band(
            likelihood, candidates, path, band_radius=band_radius,
        )

        logger.debug(
            "extract_curves_dp: curve %d  len=%d  y_bias=%.0f  "
            "x_span=%d-%d  y_mean=%.0f",
            idx, len(path), y_bias,
            path[0][0], path[-1][0],
            float(np.mean([p[1] for p in path])),
        )
        target_idx += 1

    # 4. If we still don't have enough, try without y_bias
    fallback_attempts = 0
    while len(curves) < num_curves and fallback_attempts < num_curves * 2:
        fallback_attempts += 1
        path = _extract_one_curve_dp(
            likelihood,
            candidates,
            max_jump=max_jump,
            jump_weight=jump_weight,
            curvature_weight=curvature_weight,
            min_span_ratio=min_span_ratio * 0.5,  # relax span requirement
            y_bias=None,
            y_min_bound=lower_bound_by_x,
        )
        if path is None:
            break

        idx = len(curves)
        curves[idx] = path
        exclusion_bands.append(path)
        _mask_exclusion_band(
            likelihood, candidates, path, band_radius=band_radius,
        )

    # 5. Sort curves by mean y (topmost first = lowest y in image coords)
    if len(curves) > 1:
        sorted_items = sorted(
            curves.items(),
            key=lambda kv: float(np.mean([p[1] for p in kv[1]])),
        )
        curves = {i: pixels for i, (_, pixels) in enumerate(sorted_items)}

    logger.info("extract_curves_dp: extracted %d / %d curves", len(curves), num_curves)

    debug_info: Dict[str, Any] = {
        "stroke_width": stroke_w,
        "band_radius": band_radius,
        "exclusion_bands": exclusion_bands,
        "num_attempts": attempts + fallback_attempts,
    }
    return curves, debug_info
