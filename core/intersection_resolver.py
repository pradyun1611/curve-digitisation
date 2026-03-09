"""
Intersection resolver for grayscale B/W images with crossing curves.

When multiple curves share similar gray shades and cross each other,
the standard DP tracker often confuses their identities at the crossing
point.  This module:

1. **Detects intersection zones** where two or more extracted curves
   come within a small distance of each other.
2. **Zooms into each intersection** on the *original* (un-binarised)
   grayscale image.
3. **Enhances shade differences** using aggressive local contrast
   enhancement (CLAHE with small tiles, local normalisation, unsharp
   masking) so that even subtle intensity differences between curves
   become separable.
4. **Re-traces each curve through the intersection** using a combined
   cost that balances geometric trajectory prediction (slope continuity)
   with intensity-signature matching (each curve's characteristic shade
   sampled just outside the intersection).
5. **Stitches the corrected segments** back into the full curves.

Usage
-----
Called automatically from ``extract_bw_curves`` (in ``bw_pipeline.py``)
after the DP tracker produces its initial curves.  Can also be invoked
standalone for testing::

    from core.intersection_resolver import resolve_intersections
    corrected = resolve_intersections(gray_image, curves, stroke_width=4)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    label as ndimage_label,
    uniform_filter,
)

logger = logging.getLogger(__name__)


# ====================================================================
# 1.  Detect intersection zones
# ====================================================================

def _find_intersection_zones(
    curves: Dict[int, List[Tuple[int, int]]],
    img_height: int,
    img_width: int,
    *,
    proximity_px: int = 12,
    min_zone_width: int = 8,
    merge_gap: int = 20,
    pad_x: int = 30,
) -> List[Dict[str, Any]]:
    """Find x-ranges where two or more curves are within *proximity_px*.

    Returns a list of zone dicts::

        {
            "x_start": int,        # left  boundary of the zone (padded)
            "x_end":   int,        # right boundary of the zone (padded)
            "curve_ids": List[int] # indices of the curves involved
        }
    """
    if len(curves) < 2:
        return []

    # Build per-x y-lookup for each curve (interpolated)
    curve_y_at_x: Dict[int, Dict[int, float]] = {}
    for cid, pts in curves.items():
        lookup: Dict[int, float] = {}
        # Sort by x
        sorted_pts = sorted(pts, key=lambda p: p[0])
        for px, py in sorted_pts:
            lookup[px] = float(py)
        # Simple interpolation for gaps
        if len(sorted_pts) >= 2:
            xs = np.array([p[0] for p in sorted_pts], dtype=np.float64)
            ys = np.array([p[1] for p in sorted_pts], dtype=np.float64)
            ux, inv = np.unique(xs.astype(int), return_inverse=True)
            uy = np.array(
                [np.median(ys[inv == i]) for i in range(len(ux))],
                dtype=np.float64,
            )
            x_full = np.arange(int(ux[0]), int(ux[-1]) + 1)
            y_full = np.interp(x_full, ux.astype(np.float64), uy)
            lookup = {int(x): float(y) for x, y in zip(x_full, y_full)}
        curve_y_at_x[cid] = lookup

    # For each pair of curves, find x-columns where they are close
    cids = sorted(curves.keys())
    close_columns: Dict[Tuple[int, int], List[int]] = {}

    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            ci, cj = cids[i], cids[j]
            lookup_i = curve_y_at_x[ci]
            lookup_j = curve_y_at_x[cj]
            common_xs = sorted(set(lookup_i.keys()) & set(lookup_j.keys()))
            close_xs = []
            for x in common_xs:
                if abs(lookup_i[x] - lookup_j[x]) < proximity_px:
                    close_xs.append(x)
            if close_xs:
                close_columns[(ci, cj)] = close_xs

    if not close_columns:
        return []

    # Merge close x-columns into contiguous zones per curve pair,
    # then merge across pairs for overlapping x-ranges.
    raw_zones: List[Dict[str, Any]] = []
    for (ci, cj), xs in close_columns.items():
        xs = sorted(xs)
        # Split into contiguous runs (gap > merge_gap → new zone)
        runs: List[List[int]] = []
        cur: List[int] = [xs[0]]
        for k in range(1, len(xs)):
            if xs[k] - xs[k - 1] <= merge_gap:
                cur.append(xs[k])
            else:
                runs.append(cur)
                cur = [xs[k]]
        runs.append(cur)
        for run in runs:
            if len(run) < min_zone_width:
                continue
            x0 = max(0, run[0] - pad_x)
            x1 = min(img_width - 1, run[-1] + pad_x)
            raw_zones.append({
                "x_start": x0,
                "x_end": x1,
                "curve_ids": [ci, cj],
            })

    if not raw_zones:
        return []

    # Merge overlapping zones
    raw_zones.sort(key=lambda z: z["x_start"])
    merged: List[Dict[str, Any]] = [raw_zones[0]]
    for z in raw_zones[1:]:
        prev = merged[-1]
        if z["x_start"] <= prev["x_end"] + merge_gap:
            prev["x_end"] = max(prev["x_end"], z["x_end"])
            prev["curve_ids"] = sorted(
                set(prev["curve_ids"]) | set(z["curve_ids"])
            )
        else:
            merged.append(z)

    logger.info(
        "Intersection resolver: found %d zone(s) across %d curve pair(s)",
        len(merged), len(close_columns),
    )
    for z in merged:
        logger.debug(
            "  zone x=[%d, %d]  curves=%s",
            z["x_start"], z["x_end"], z["curve_ids"],
        )

    return merged


# ====================================================================
# 2.  Enhance shade differences in a zoomed ROI
# ====================================================================

def _enhance_roi(
    gray_roi: np.ndarray,
    *,
    clahe_clip: float = 12.0,
    clahe_tile: int = 4,
    local_norm_size: int = 15,
    unsharp_sigma: float = 1.5,
    unsharp_strength: float = 2.5,
    gamma: float = 0.0,
) -> np.ndarray:
    """Apply *very* aggressive local contrast enhancement to a small ROI.

    Pipeline (all stages stack):
      1. CLAHE pass 1 — small tiles, high clip  (local histogram EQ)
      2. CLAHE pass 2 — larger tiles, moderate clip  (broader EQ)
      3. Gamma correction — auto-selected to push mid-tones apart
      4. Local normalisation — subtract local mean / divide local std
      5. Unsharp masking — sharpen stroke edges
      6. Percentile stretch — map the 2nd–98th percentile range to [0,1]

    Returns a float32 image in [0, 1] with maximised shade separation.
    """
    roi_u8 = np.clip(gray_roi, 0, 255).astype(np.uint8)
    h, w = roi_u8.shape

    # --- Step 1 & 2: Dual CLAHE ---
    try:
        import cv2

        # Pass 1 — fine-grained, high clip (local contrast boost)
        tile1 = max(2, clahe_tile)
        c1 = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(tile1, tile1))
        pass1 = c1.apply(roi_u8)

        # Pass 2 — coarser tiles, moderate clip (mid-tone separation)
        tile2 = max(4, clahe_tile * 2)
        c2 = cv2.createCLAHE(clipLimit=max(4.0, clahe_clip * 0.5),
                              tileGridSize=(tile2, tile2))
        pass2 = c2.apply(pass1)
        roi_f = pass2.astype(np.float32)
    except ImportError:
        # Fallback: manual percentile stretch
        lo = float(np.percentile(roi_u8, 2))
        hi = float(np.percentile(roi_u8, 98))
        rng = max(hi - lo, 1.0)
        roi_f = (roi_u8.astype(np.float32) - lo) / rng * 255.0
        roi_f = np.clip(roi_f, 0, 255)

    # --- Step 3: Gamma correction ---
    # Auto gamma: if the image has a narrow mid-range, pick gamma to
    # spread it.  gamma < 1 → brighten darks, gamma > 1 → darken lights.
    if gamma <= 0.0:
        med = float(np.median(roi_f))
        # Target: push median toward 128 to maximise dynamic range
        if 10 < med < 245:
            gamma = math.log(128.0 / 255.0) / math.log(med / 255.0)
            gamma = max(0.3, min(gamma, 3.0))
        else:
            gamma = 1.0

    roi_norm = roi_f / 255.0
    roi_norm = np.power(np.clip(roi_norm, 0, 1), gamma)
    roi_f = roi_norm * 255.0

    # --- Step 4: Local normalisation ---
    local_mean = uniform_filter(roi_f, size=local_norm_size)
    diff = roi_f - local_mean
    local_var = uniform_filter(diff ** 2, size=local_norm_size)
    local_std = np.sqrt(np.clip(local_var, 1.0, None))
    normalised = diff / local_std  # zero-mean, unit-variance locally

    # Map back to [0, 1] via sigmoid-like scaling
    normalised = np.clip(normalised, -3, 3)
    normalised = (normalised + 3.0) / 6.0  # [0, 1]

    # --- Step 5: Unsharp mask ---
    blurred = gaussian_filter(normalised, sigma=unsharp_sigma)
    sharpened = normalised + unsharp_strength * (normalised - blurred)

    # --- Step 6: Percentile stretch to [0, 1] ---
    lo = float(np.percentile(sharpened, 2))
    hi = float(np.percentile(sharpened, 98))
    rng = max(hi - lo, 1e-6)
    sharpened = (sharpened - lo) / rng
    sharpened = np.clip(sharpened, 0, 1).astype(np.float32)

    return sharpened


def _sample_intensity_signature(
    gray: np.ndarray,
    curve_pts: List[Tuple[int, int]],
    zone_x_start: int,
    zone_x_end: int,
    *,
    sample_width: int = 25,
    sample_radius: int = 3,
    binary: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Sample the intensity signature of a curve just outside a zone.

    Uses **median + IQR** (inter-quartile range) instead of mean + std
    for robustness against outlier contamination.

    **New**: if the combined left+right sample has a very high IQR
    (indicating the curve path is already interchanged), we fall back to
    sampling from each side independently and picking the more consistent
    side.  This prevents contamination from a previous crossing.

    Returns (median_intensity, iqr_half) on the original grayscale.
    *iqr_half* = (Q3 - Q1) / 2  (analogous to 1-sigma spread).
    """
    h, w = gray.shape
    pts_sorted = sorted(curve_pts, key=lambda p: p[0])

    def _collect(pts_list: List[Tuple[int, int]], radius: int) -> List[float]:
        vals: List[float] = []
        for px, py in pts_list:
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if binary is not None and not binary[ny, nx]:
                            continue
                        vals.append(float(gray[ny, nx]))
        return vals

    def _robust_stats(values: List[float]) -> Tuple[float, float]:
        """Compute (median, iqr_half) with outlier rejection."""
        if not values:
            return (128.0, 30.0)
        arr = np.array(values, dtype=np.float64)
        q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        iqr = q3 - q1
        if iqr > 0:
            lo_fence = q1 - 1.5 * iqr
            hi_fence = q3 + 1.5 * iqr
            filtered = arr[(arr >= lo_fence) & (arr <= hi_fence)]
            if len(filtered) >= 3:
                arr = filtered
        med = float(np.median(arr))
        q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        return (med, max((q3 - q1) / 2.0, 5.0))

    # --- Collect sample points just outside the zone ---
    left_pts = [
        (px, py) for px, py in pts_sorted
        if zone_x_start - sample_width <= px < zone_x_start
    ]
    right_pts = [
        (px, py) for px, py in pts_sorted
        if zone_x_end < px <= zone_x_end + sample_width
    ]

    # If narrow windows are empty, widen
    if not left_pts:
        all_left = [p for p in pts_sorted if p[0] < zone_x_start]
        if all_left:
            left_pts = all_left[-sample_width:]
    if not right_pts:
        all_right = [p for p in pts_sorted if p[0] > zone_x_end]
        if all_right:
            right_pts = all_right[:sample_width]

    # If BOTH sides are empty, use extremities
    if not left_pts and not right_pts:
        n_ext = min(sample_width, len(pts_sorted) // 4, 30)
        if n_ext >= 3:
            left_pts = pts_sorted[:n_ext]
            right_pts = pts_sorted[-n_ext:]

    if not left_pts and not right_pts:
        return (128.0, 30.0)

    # --- Strategy: try combined, then fall back to single-side if noisy ---
    left_vals = _collect(left_pts, sample_radius) if left_pts else []
    right_vals = _collect(right_pts, sample_radius) if right_pts else []

    # Fallback: exact pixel locations (no neighbourhood) if on-mask filtering
    # removed everything
    if not left_vals and left_pts:
        for px, py in left_pts:
            if 0 <= py < h and 0 <= px < w:
                left_vals.append(float(gray[py, px]))
    if not right_vals and right_pts:
        for px, py in right_pts:
            if 0 <= py < h and 0 <= px < w:
                right_vals.append(float(gray[py, px]))

    combined = left_vals + right_vals
    if not combined:
        return (128.0, 30.0)

    combined_med, combined_iqr = _robust_stats(combined)

    # If IQR is reasonable, use the combined result
    IQR_HIGH_THRESHOLD = 40.0
    if combined_iqr <= IQR_HIGH_THRESHOLD:
        return (combined_med, combined_iqr)

    # IQR is too high — the curve path is likely already interchanged,
    # so one side has different shade than the other.  Pick the more
    # consistent side.
    left_med, left_iqr = _robust_stats(left_vals) if left_vals else (128.0, 999.0)
    right_med, right_iqr = _robust_stats(right_vals) if right_vals else (128.0, 999.0)

    logger.debug(
        "    Signature split: left=(%.1f, %.1f) right=(%.1f, %.1f) "
        "combined=(%.1f, %.1f)",
        left_med, left_iqr, right_med, right_iqr, combined_med, combined_iqr,
    )

    # Pick the side with lower IQR (more consistent = more likely correct)
    if left_iqr < right_iqr and left_vals:
        if left_iqr < IQR_HIGH_THRESHOLD:
            return (left_med, left_iqr)
    if right_iqr < left_iqr and right_vals:
        if right_iqr < IQR_HIGH_THRESHOLD:
            return (right_med, right_iqr)

    # If both sides are noisy, try tighter sampling (radius=1)
    if sample_radius > 1:
        tight_left = _collect(left_pts, 1) if left_pts else []
        tight_right = _collect(right_pts, 1) if right_pts else []
        tight_l_med, tight_l_iqr = _robust_stats(tight_left) if tight_left else (128.0, 999.0)
        tight_r_med, tight_r_iqr = _robust_stats(tight_right) if tight_right else (128.0, 999.0)
        best_med, best_iqr = combined_med, combined_iqr
        for m, i in [(tight_l_med, tight_l_iqr), (tight_r_med, tight_r_iqr)]:
            if i < best_iqr:
                best_med, best_iqr = m, i
        if best_iqr < combined_iqr * 0.7:
            return (best_med, best_iqr)

    # Final fallback: return combined (noisy but best we have)
    return (combined_med, combined_iqr)


# ====================================================================
# 2b. Global shade cluster detection + per-curve assignment
# ====================================================================

def _detect_shade_clusters(
    gray: np.ndarray,
    binary: np.ndarray,
    n_clusters: int,
) -> List[float]:
    """Detect the dominant shade levels present in the binary mask.

    Finds *all* distinct shade peaks (separated by ≥ MIN_SEPARATION in
    intensity) from the histogram of on-mask pixel intensities.  Returns
    at least *n_clusters* centres so the caller always has enough
    candidates for assignment.

    Returns a sorted list of cluster centres (ascending shade, i.e.
    darkest first).
    """
    h, w = gray.shape
    # Collect all on-mask intensities
    mask_pixels = gray[binary > 0].astype(np.float64)
    if len(mask_pixels) < 10:
        # Not enough data — return evenly spaced defaults
        return [float(i * 255 / (n_clusters + 1)) for i in range(1, n_clusters + 1)]

    # Compute histogram
    hist, bin_edges = np.histogram(mask_pixels, bins=64, range=(0, 255))
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Smooth to find robust peaks
    from scipy.ndimage import gaussian_filter1d
    hist_smooth = gaussian_filter1d(hist.astype(np.float64), sigma=1.5)

    # Find peaks: local maxima
    peaks: List[Tuple[float, float]] = []  # (height, centre)
    for i in range(1, len(hist_smooth) - 1):
        if hist_smooth[i] > hist_smooth[i - 1] and hist_smooth[i] > hist_smooth[i + 1]:
            peaks.append((float(hist_smooth[i]), float(bin_centres[i])))

    # Sort by height (most populated first)
    peaks.sort(reverse=True)

    # Select ALL peaks that are at least MIN_SEPARATION apart
    # (not capped at n_clusters — the caller will choose which to use)
    MIN_SEPARATION = 20.0
    MIN_HEIGHT_RATIO = 0.05  # ignore peaks < 5% of tallest
    tallest = peaks[0][0] if peaks else 1.0
    selected: List[float] = []
    for height, centre in peaks:
        if height < tallest * MIN_HEIGHT_RATIO:
            continue  # too small — noise
        if all(abs(centre - s) >= MIN_SEPARATION for s in selected):
            selected.append(centre)

    # If we didn't find enough peaks, fill with histogram percentiles
    while len(selected) < n_clusters:
        pct = (len(selected) + 1) / (n_clusters + 1) * 100
        selected.append(float(np.percentile(mask_pixels, pct)))

    selected.sort()
    return selected


def _assign_curves_to_clusters(
    gray: np.ndarray,
    binary: np.ndarray,
    curves: Dict[int, List[Tuple[int, int]]],
    zone_x_start: int,
    zone_x_end: int,
    cluster_centres: List[float],
    per_curve_signatures: Dict[int, Tuple[float, float]],
) -> Dict[int, Tuple[float, float]]:
    """Assign each curve to the best-matching shade cluster.

    For each curve, collects the outside-zone on-mask pixel intensities
    and picks the cluster whose centre is closest to the *mode* of those
    intensities.  Falls back to the per-curve signature if the curve has
    insufficient data.

    For curves with very noisy signatures (iqr_half > 40), the cluster
    assignment is preferred.  For curves with clean signatures, the
    original signature is kept.

    Returns a dict of cid → (cluster_centre, cluster_spread).
    """
    h, w = gray.shape
    IQR_TRUST_THRESHOLD = 40.0

    assigned: Dict[int, Tuple[float, float]] = {}
    used_clusters: Dict[float, int] = {}  # cluster_centre → cid (to avoid duplicates)

    # First pass: assign curves with clean signatures directly
    for cid, (med, iqr) in per_curve_signatures.items():
        if iqr <= IQR_TRUST_THRESHOLD:
            # Find nearest cluster
            best_cluster = min(cluster_centres, key=lambda c: abs(c - med))
            assigned[cid] = (best_cluster, iqr)
            used_clusters[best_cluster] = cid

    # Second pass: assign noisy curves to the remaining clusters
    noisy_cids = [
        cid for cid in per_curve_signatures
        if cid not in assigned
    ]

    if noisy_cids:
        remaining_clusters = [c for c in cluster_centres if c not in used_clusters]

        for cid in noisy_cids:
            pts = curves.get(cid, [])
            # Collect outside-zone on-mask intensities
            vals: List[float] = []
            for px, py in pts:
                if px < zone_x_start or px > zone_x_end:
                    if 0 <= py < h and 0 <= px < w and binary[py, px]:
                        vals.append(float(gray[py, px]))

            if vals and remaining_clusters:
                arr = np.array(vals, dtype=np.float64)
                # Use the mode (most frequent shade) — find which cluster
                # has the most pixels within ±20 of its centre
                best_cluster = None
                best_count = -1
                for cc in remaining_clusters:
                    count = int(np.sum(np.abs(arr - cc) < 25))
                    if count > best_count:
                        best_count = count
                        best_cluster = cc
                if best_cluster is not None:
                    assigned[cid] = (best_cluster, 15.0)
                    remaining_clusters.remove(best_cluster)
                    logger.debug(
                        "    Cluster assigned curve %d → shade=%.1f "
                        "(was noisy med=%.1f iqr=%.1f)",
                        cid, best_cluster,
                        per_curve_signatures[cid][0],
                        per_curve_signatures[cid][1],
                    )
                else:
                    # Fallback: use original noisy signature
                    assigned[cid] = per_curve_signatures[cid]
            elif remaining_clusters:
                # No outside-zone pixels — assign the nearest remaining cluster
                orig_med = per_curve_signatures[cid][0]
                best_cluster = min(remaining_clusters, key=lambda c: abs(c - orig_med))
                assigned[cid] = (best_cluster, 20.0)
                remaining_clusters.remove(best_cluster)
                logger.debug(
                    "    Cluster fallback curve %d → shade=%.1f (no data)",
                    cid, best_cluster,
                )
            else:
                assigned[cid] = per_curve_signatures[cid]

    return assigned


# ====================================================================
# 3.  Intensity-guided DP re-tracing through an intersection zone
# ====================================================================

def _retrace_through_zone(
    gray: np.ndarray,
    enhanced: np.ndarray,
    binary: np.ndarray,
    curve_pts: List[Tuple[int, int]],
    intensity_mean: float,
    intensity_std: float,
    zone_x_start: int,
    zone_x_end: int,
    zone_y_start: int,
    zone_y_end: int,
    *,
    approach_pts: int = 15,
    max_jump: int = 5,
    intensity_weight: float = 0.85,
    geometry_weight: float = 0.15,
    curvature_weight: float = 0.10,
    other_signatures: Optional[List[Tuple[float, float]]] = None,
) -> Optional[List[Tuple[int, int]]]:
    """Re-trace a single curve through an intersection zone.

    Tries **left-to-right** first.  If that fails (lost track), tries
    **right-to-left**.  If both fail, returns None.
    """
    # Try left-to-right
    result = _retrace_directional(
        gray, enhanced, binary, curve_pts,
        intensity_mean, intensity_std,
        zone_x_start, zone_x_end, zone_y_start, zone_y_end,
        approach_pts=approach_pts, max_jump=max_jump,
        intensity_weight=intensity_weight,
        geometry_weight=geometry_weight,
        curvature_weight=curvature_weight,
        other_signatures=other_signatures,
        reverse=False,
    )
    if result is not None:
        return result

    # Try right-to-left (may work better when there are more/cleaner
    # approach points on the right side)
    logger.debug("_retrace: LTR failed, trying RTL")
    result = _retrace_directional(
        gray, enhanced, binary, curve_pts,
        intensity_mean, intensity_std,
        zone_x_start, zone_x_end, zone_y_start, zone_y_end,
        approach_pts=approach_pts, max_jump=max_jump,
        intensity_weight=intensity_weight,
        geometry_weight=geometry_weight,
        curvature_weight=curvature_weight,
        other_signatures=other_signatures,
        reverse=True,
    )
    return result


def _retrace_directional(
    gray: np.ndarray,
    enhanced: np.ndarray,
    binary: np.ndarray,
    curve_pts: List[Tuple[int, int]],
    intensity_mean: float,
    intensity_std: float,
    zone_x_start: int,
    zone_x_end: int,
    zone_y_start: int,
    zone_y_end: int,
    *,
    approach_pts: int = 15,
    max_jump: int = 5,
    intensity_weight: float = 0.85,
    geometry_weight: float = 0.15,
    curvature_weight: float = 0.10,
    other_signatures: Optional[List[Tuple[float, float]]] = None,
    reverse: bool = False,
) -> Optional[List[Tuple[int, int]]]:
    """Core single-direction DP re-trace.

    When *reverse* is True, traces right-to-left (swaps entry/exit).
    """
    h, w = gray.shape
    pts_sorted = sorted(curve_pts, key=lambda p: p[0])

    # --- Estimate approach trajectory (slope) ---
    left_pts = [
        (px, py) for px, py in pts_sorted
        if zone_x_start - approach_pts * 2 <= px < zone_x_start
    ]
    right_pts = [
        (px, py) for px, py in pts_sorted
        if zone_x_end < px <= zone_x_end + approach_pts * 2
    ]

    # Entry/exit slopes
    entry_slope = 0.0
    entry_y: Optional[float] = None
    if len(left_pts) >= 2:
        lp = np.array(left_pts[-approach_pts:], dtype=np.float64)
        if len(lp) >= 2:
            dx = lp[-1, 0] - lp[0, 0]
            if dx > 0:
                entry_slope = (lp[-1, 1] - lp[0, 1]) / dx
            entry_y = float(lp[-1, 1])

    exit_slope = 0.0
    exit_y: Optional[float] = None
    if len(right_pts) >= 2:
        rp = np.array(right_pts[:approach_pts], dtype=np.float64)
        if len(rp) >= 2:
            dx = rp[-1, 0] - rp[0, 0]
            if dx > 0:
                exit_slope = (rp[-1, 1] - rp[0, 1]) / dx
            exit_y = float(rp[0, 1])

    # In reverse mode, swap entry/exit
    if reverse:
        entry_y, exit_y = exit_y, entry_y
        entry_slope, exit_slope = -exit_slope, -entry_slope

    # If we have no entry info at all, use the curve's points inside zone
    if entry_y is None:
        zone_pts = [
            (px, py) for px, py in pts_sorted
            if zone_x_start <= px <= zone_x_end
        ]
        if zone_pts:
            if reverse:
                entry_y = float(zone_pts[-1][1])
            else:
                entry_y = float(zone_pts[0][1])
        else:
            return None

    # --- Precompute enhanced-image signature for this curve ---
    enh_mean = 0.5
    if enhanced is not None:
        enh_vals: List[float] = []
        sample_left = [
            (px, py) for px, py in pts_sorted
            if zone_x_start - 20 <= px < zone_x_start
        ]
        sample_right = [
            (px, py) for px, py in pts_sorted
            if zone_x_end < px <= zone_x_end + 20
        ]
        for px, py in (sample_left + sample_right):
            if 0 <= py < h and 0 <= px < w:
                enh_vals.append(float(enhanced[py, px]))
        if enh_vals:
            enh_mean = float(np.median(enh_vals))

    # --- Precompute vertical gradient ---
    grad_y = np.abs(np.diff(gray.astype(np.float32), axis=0))
    grad_y = np.vstack([grad_y, grad_y[-1:, :]])

    # --- Build x-column sequence (forward or reverse) ---
    if reverse:
        x_cols = list(range(zone_x_end, zone_x_start - 1, -1))
    else:
        x_cols = list(range(zone_x_start, zone_x_end + 1))

    if len(x_cols) < 2:
        return None

    INF = float("inf")
    y_window = max(max_jump * 4, int((zone_y_end - zone_y_start) * 0.8), 30)
    norm_std = max(intensity_std, 5.0)

    def _intensity_cost(y: int, x: int) -> float:
        if not (0 <= y < h and 0 <= x < w):
            return 2.0
        pixel_val = float(gray[y, x])
        own_dist = abs(pixel_val - intensity_mean)
        own_cost = own_dist / norm_std

        enh_cost = 0.0
        if enhanced is not None:
            enh_val = float(enhanced[y, x])
            enh_cost = abs(enh_val - enh_mean) * 2.0

        repulsion = 0.0
        if other_signatures:
            for o_mean, o_std in other_signatures:
                other_dist = abs(pixel_val - o_mean)
                if other_dist < own_dist:
                    ratio = 1.0 - other_dist / max(own_dist, 1.0)
                    repulsion += 0.7 * ratio
                elif other_dist < own_dist * 1.3:
                    repulsion += 0.15

        return intensity_weight * (own_cost + enh_cost * 0.4 + repulsion)

    def _binary_bonus(y: int, x: int) -> float:
        if not (0 <= y < h and 0 <= x < w):
            return 0.0
        if not binary[y, x]:
            return 0.0
        pixel_val = float(gray[y, x])
        if abs(pixel_val - intensity_mean) < norm_std * 2.5:
            return -0.5
        return 0.15

    # --- DP forward pass along x_cols ---
    cost_prev: Dict[int, float] = {}
    back_ptr: Dict[int, Dict[int, int]] = {}
    slope_prev: Dict[int, float] = {}

    # Initialize at first column
    x_first = x_cols[0]
    pred_y = entry_y
    for y in range(max(0, int(pred_y - y_window)), min(h, int(pred_y + y_window))):
        int_cost = _intensity_cost(y, x_first)
        bin_bonus = _binary_bonus(y, x_first)
        geo_cost = geometry_weight * abs(y - pred_y) / max(y_window, 1)
        cost_prev[y] = int_cost + geo_cost + bin_bonus
        slope_prev[y] = entry_slope

    if not cost_prev:
        return None

    MAX_CANDIDATES = 40
    total_cols = len(x_cols)

    for ci in range(1, total_cols):
        x = x_cols[ci]
        cost_cur: Dict[int, float] = {}
        back_cur: Dict[int, int] = {}
        slope_cur: Dict[int, float] = {}

        frac = ci / max(total_cols - 1, 1)
        if exit_y is not None and entry_y is not None:
            predicted_center = entry_y + frac * (exit_y - entry_y)
        elif entry_y is not None:
            predicted_center = entry_y + entry_slope * ci
        else:
            predicted_center = (zone_y_start + zone_y_end) / 2.0

        y_lo = max(0, int(predicted_center - y_window))
        y_hi = min(h, int(predicted_center + y_window))

        for y in range(y_lo, y_hi):
            if y < 0 or y >= h or x < 0 or x >= w:
                continue

            int_cost = _intensity_cost(y, x)
            bin_bonus = _binary_bonus(y, x)

            best_cost = INF
            best_py = -1
            for py, pc in cost_prev.items():
                dy = abs(y - py)
                if dy > max_jump:
                    continue

                jump_cost = geometry_weight * 0.3 * dy
                old_slope = slope_prev.get(py, entry_slope)
                predicted = py + old_slope
                curv_cost = curvature_weight * abs(y - predicted)

                edge_cost = 0.0
                if dy > 1 and 0 <= x < w:
                    y_lo_e = min(y, py)
                    y_hi_e = max(y, py)
                    if y_hi_e < h:
                        max_grad = float(
                            grad_y[y_lo_e:y_hi_e + 1, x].max()
                        )
                        edge_cost = 0.15 * max_grad / 30.0

                total = pc + int_cost + jump_cost + curv_cost + bin_bonus + edge_cost
                if total < best_cost:
                    best_cost = total
                    best_py = py

            if best_py >= 0:
                cost_cur[y] = best_cost
                back_cur[y] = best_py
                dy_actual = y - best_py
                old_slope = slope_prev.get(best_py, entry_slope)
                slope_cur[y] = 0.7 * old_slope + 0.3 * dy_actual

        back_ptr[x] = back_cur
        cost_prev = cost_cur
        slope_prev = slope_cur

        # Prune
        if len(cost_prev) > MAX_CANDIDATES:
            sorted_ys = sorted(cost_prev, key=cost_prev.get)  # type: ignore[arg-type]
            keep = set(sorted_ys[:MAX_CANDIDATES])
            cost_prev = {y: c for y, c in cost_prev.items() if y in keep}
            slope_prev = {y: s for y, s in slope_prev.items() if y in keep}

        if not cost_prev:
            direction = "RTL" if reverse else "LTR"
            logger.debug(
                "_retrace(%s): lost track at x=%d (col %d/%d)",
                direction, x, ci, total_cols,
            )
            return None

    if not cost_prev:
        return None

    # Apply exit-y preference
    if exit_y is not None:
        for y in cost_prev:
            cost_prev[y] += geometry_weight * abs(y - exit_y) * 0.5

    best_y = min(cost_prev, key=cost_prev.get)  # type: ignore[arg-type]

    # Back-trace through the x_cols sequence
    x_last = x_cols[-1]
    path: List[Tuple[int, int]] = [(x_last, best_y)]
    cy = best_y
    for ci in range(total_cols - 1, 0, -1):
        x = x_cols[ci]
        if x in back_ptr and cy in back_ptr[x]:
            cy = back_ptr[x][cy]
            path.append((x_cols[ci - 1], cy))
        else:
            break

    path.reverse()

    # In reverse mode, path is right-to-left — already has correct (x, y) pairs
    # but we need to ensure it's sorted left-to-right for output
    path.sort(key=lambda p: p[0])

    if len(path) < max(3, (zone_x_end - zone_x_start) // 3):
        return None

    return path


# ====================================================================
# 4.  Vertical ROI bounds for a zone
# ====================================================================

def _zone_y_bounds(
    curves: Dict[int, List[Tuple[int, int]]],
    zone: Dict[str, Any],
    img_height: int,
    pad_y: int = 40,
) -> Tuple[int, int]:
    """Compute the vertical extent of an intersection zone."""
    cids = zone["curve_ids"]
    x0, x1 = zone["x_start"], zone["x_end"]

    all_ys: List[int] = []
    for cid in cids:
        if cid not in curves:
            continue
        for px, py in curves[cid]:
            if x0 - 20 <= px <= x1 + 20:
                all_ys.append(py)

    if not all_ys:
        return (0, img_height - 1)

    y_min = max(0, min(all_ys) - pad_y)
    y_max = min(img_height - 1, max(all_ys) + pad_y)
    return (y_min, y_max)


# ====================================================================
# 4b.  Shade-guided curve re-extraction
# ====================================================================

def _shade_guided_reextract(
    gray: np.ndarray,
    binary: np.ndarray,
    curves: Dict[int, List[Tuple[int, int]]],
    cluster_centres: List[float],
    signatures: Dict[int, Tuple[float, float]],
    zone_x_start: int,
    zone_x_end: int,
    *,
    shade_tolerance: float = 0.0,
    max_gap: int = 3,
) -> Dict[int, Optional[List[Tuple[int, int]]]]:
    """Re-extract curves from shade-specific binary masks.

    For each curve (identified by its assigned shade cluster), creates a
    binary mask containing only on-mask pixels within *shade_tolerance*
    of the cluster centre, then uses a column-wise trace to extract a
    clean single-curve path **within the intersection zone only**.

    This approach bypasses the intersection problem entirely: each shade
    mask contains only one curve's pixels.

    Returns only the zone segment [zone_x_start, zone_x_end], not attempting
    to extend beyond the zone. Caller will stitch this segment into the
    full curve.

    Parameters
    ----------
    gray : (H, W) uint8 grayscale image
    binary : (H, W) bool mask
    curves : original curves (for zone y-bounds estimation only)
    cluster_centres : detected shade levels (sorted ascending)
    signatures : cid → (cluster_centre, spread) from assignment
    zone_x_start, zone_x_end : zone boundaries
    shade_tolerance : radius around each cluster centre to include.
        0 = auto-compute as half the minimum gap between clusters.
    max_gap : max consecutive missing x-columns before breaking the trace

    Returns
    -------
    Dict[cid, List[(x,y)] or None] — re-extracted zone segments ONLY
    """
    h, w = gray.shape

    # Auto-compute shade tolerance: half the min gap between clusters
    if shade_tolerance <= 0.0:
        if len(cluster_centres) >= 2:
            gaps = [
                cluster_centres[i + 1] - cluster_centres[i]
                for i in range(len(cluster_centres) - 1)
            ]
            shade_tolerance = max(min(gaps) / 2.0, 10.0)
        else:
            shade_tolerance = 25.0

    results: Dict[int, Optional[List[Tuple[int, int]]]] = {}

    for cid, (centre, spread) in signatures.items():
        if cid not in curves:
            results[cid] = None
            continue

        pts_sorted = sorted(curves[cid], key=lambda p: p[0])

        # --- Create shade-specific binary mask ---
        shade_mask = (
            binary
            & (gray >= max(0, centre - shade_tolerance))
            & (gray <= min(255, centre + shade_tolerance))
        )

        # --- Trace through the zone using shade mask ---
        # Find the entry point: first shade pixel at or after zone_x_start
        path: List[Tuple[int, int]] = []
        last_y: Optional[float] = None

        for x in range(zone_x_start, zone_x_end + 1):
            if x < 0 or x >= w:
                continue

            # Find on-mask shade pixels in this column
            on_pixels = [
                y for y in range(h)
                if shade_mask[y, x]
            ]

            if not on_pixels:
                # No pixels in this column — try to predict
                # but only if we have recent data
                if last_y is not None and len(path) > 0:
                    # Use the last y position
                    if len(path) < max_gap * 2:  # short gap — predict
                        path.append((x, int(round(last_y))))
                # else: skip this column
                continue

            # We have pixels — find the best one
            if last_y is None:
                # First column — pick median y (centre of curve)
                best_y = int(np.median(on_pixels))
            else:
                # Pick closest to expected y for continuity
                best_y = min(on_pixels, key=lambda y: abs(y - last_y))

            path.append((x, best_y))
            last_y = float(best_y)

        # Check if we got enough points
        if len(path) < max(5, (zone_x_end - zone_x_start) // 4):
            logger.debug(
                "  shade_reextract curve %d: only %d pts (too few), discarding",
                cid, len(path),
            )
            results[cid] = None
        else:
            results[cid] = path
            logger.debug(
                "  shade_reextract curve %d (shade=%.0f+/-%.0f): %d pts in zone",
                cid, centre, shade_tolerance, len(path),
            )

    return results


# ====================================================================
# 5.  Public entry point
# ====================================================================

def resolve_intersections(
    gray_image: np.ndarray,
    binary: np.ndarray,
    curves: Dict[int, List[Tuple[int, int]]],
    *,
    stroke_width: int = 4,
    proximity_px: int = 0,
    intensity_weight: float = 0.85,
    geometry_weight: float = 0.15,
    curvature_weight: float = 0.10,
    max_jump: int = 0,
    debug_dir: str = "",
) -> Dict[int, List[Tuple[int, int]]]:
    """Resolve curve crossings in grayscale images using shade analysis.

    This is the main entry point called from ``extract_bw_curves``.

    Parameters
    ----------
    gray_image : (H, W) uint8
        Original grayscale image (cropped to plot area).
    binary : (H, W) bool
        Cleaned binary mask (same coordinate frame as gray_image).
    curves : Dict[int, List[(x, y)]]
        Initial curve paths from the DP tracker (plot-area local coords).
    stroke_width : int
        Estimated stroke width in pixels.
    proximity_px : int
        Max y-distance to consider curves as "intersecting".
        0 = auto-compute from stroke_width.
    intensity_weight : float
        Weight for the intensity-match cost in the DP re-tracer (dominant).
    geometry_weight : float
        Weight for the geometric/trajectory cost.
    curvature_weight : float
        Weight for the curvature (2nd derivative) cost.
    max_jump : int
        Max y-change per x-column in the re-tracer.  0 = auto.
    debug_dir : str
        If non-empty, save debug visualisations here.

    Returns
    -------
    Dict[int, List[(x, y)]]
        Corrected curves with intersection segments re-traced.
    """
    if len(curves) < 2:
        return curves

    h, w = gray_image.shape[:2]

    # Auto-compute parameters — tighter max_jump to prevent hopping
    if proximity_px <= 0:
        proximity_px = max(stroke_width * 3, 12)
    if max_jump <= 0:
        max_jump = max(stroke_width + 2, 5)

    # 1. Detect intersection zones
    zones = _find_intersection_zones(
        curves, h, w,
        proximity_px=proximity_px,
        min_zone_width=max(5, stroke_width),
        merge_gap=max(20, stroke_width * 5),
        pad_x=max(30, stroke_width * 8),
    )

    if not zones:
        logger.info("Intersection resolver: no intersection zones found")
        return curves

    logger.info(
        "Intersection resolver: processing %d zone(s) with %d curves",
        len(zones), len(curves),
    )

    # 2. Enhance the full image once — very aggressive contrast boost
    enhanced_full = _enhance_roi(
        gray_image,
        clahe_clip=12.0,
        clahe_tile=4,
        local_norm_size=max(11, stroke_width * 3),
        unsharp_sigma=1.5,
        unsharp_strength=2.5,
    )

    # 3. Process each zone
    corrected_curves = {cid: list(pts) for cid, pts in curves.items()}

    for zi, zone in enumerate(zones):
        x0, x1 = zone["x_start"], zone["x_end"]
        y0, y1 = _zone_y_bounds(curves, zone, h)
        cids = zone["curve_ids"]

        logger.debug(
            "Zone %d: x=[%d,%d] y=[%d,%d] curves=%s",
            zi, x0, x1, y0, y1, cids,
        )

        # Sample intensity signatures for each curve in this zone
        signatures: Dict[int, Tuple[float, float]] = {}
        for cid in cids:
            if cid not in corrected_curves:
                continue
            mean_i, std_i = _sample_intensity_signature(
                gray_image,
                corrected_curves[cid],
                x0, x1,
                sample_width=max(25, (x1 - x0) // 2),
                sample_radius=max(2, stroke_width // 2),
                binary=binary,
            )
            signatures[cid] = (mean_i, std_i)
            logger.debug(
                "  Curve %d intensity: median=%.1f iqr_half=%.1f",
                cid, mean_i, std_i,
            )

        # If any signature is noisy (iqr > 40), run global cluster
        # detection and reassign noisy curves to the correct cluster.
        has_noisy = any(s[1] > 40.0 for s in signatures.values())
        if has_noisy and len(cids) >= 2:
            cluster_centres = _detect_shade_clusters(
                gray_image, binary, n_clusters=len(curves),
            )
            logger.debug(
                "  Shade clusters detected: %s",
                [f"{c:.0f}" for c in cluster_centres],
            )
            signatures = _assign_curves_to_clusters(
                gray_image, binary,
                corrected_curves, x0, x1,
                cluster_centres, signatures,
            )
            for cid in cids:
                if cid in signatures:
                    logger.debug(
                        "  Curve %d final signature: shade=%.1f spread=%.1f",
                        cid, signatures[cid][0], signatures[cid][1],
                    )

        # Check if curves actually have different shades
        sig_means = [s[0] for s in signatures.values()]
        if len(sig_means) >= 2:
            max_diff = max(sig_means) - min(sig_means)
        else:
            max_diff = 0.0

        if max_diff < 8:
            logger.debug(
                "  Zone %d: shade diff=%.1f too small, "
                "using geometry-only DP re-trace",
                zi, max_diff,
            )
            zone_int_w = 0.15
            zone_geo_w = 0.85
            use_shade_reextract = False
        else:
            zone_int_w = intensity_weight
            zone_geo_w = geometry_weight
            use_shade_reextract = True
            logger.debug(
                "  Zone %d: shade diff=%.1f — using shade-guided re-extraction",
                zi, max_diff,
            )

        # ── PRIMARY: shade-guided re-extraction ──────────────────────
        # Works by creating a binary mask per shade cluster and tracing
        # each curve on its own clean mask.  This is the most robust
        # approach when shade separation is adequate.
        if use_shade_reextract:
            # Determine cluster centres (may already have been computed
            # in the noisy-signature branch above)
            if not has_noisy:
                # No noisy signatures — compute clusters now
                cluster_centres = _detect_shade_clusters(
                    gray_image, binary, n_clusters=len(curves),
                )
                logger.debug(
                    "  Shade clusters detected: %s",
                    [f"{c:.0f}" for c in cluster_centres],
                )
                # Reassign signatures to nearest cluster centre
                for cid in list(signatures.keys()):
                    med, iqr = signatures[cid]
                    best_c = min(cluster_centres, key=lambda c: abs(c - med))
                    signatures[cid] = (best_c, iqr)

            # --- Unused-cluster check ---
            # If more cluster centres than zone curves, some clusters
            # may go unused.  ONLY reassign if the side-sample STRONGLY
            # prefers an unused cluster (diff > 15).  This avoids over-
            # correction when the zone assignment is already reasonable.
            used_in_zone = set(s[0] for s in signatures.values())
            unused_clusters = [
                c for c in cluster_centres
                if c not in used_in_zone
            ]

            if unused_clusters and len(cids) >= 2:
                sw_half = max(25, (x1 - x0) // 2)
                for cid in cids:
                    if cid not in signatures:
                        continue
                    current_centre, current_iqr = signatures[cid]
                    pts = corrected_curves.get(cid, [])
                    pts_sorted = sorted(pts, key=lambda p: p[0])

                    # Sample LEFT-only and RIGHT-only
                    for side_name, side_pts in [
                        ("left", [p for p in pts_sorted if p[0] < x0][-sw_half:]),
                        ("right", [p for p in pts_sorted if p[0] > x1][:sw_half]),
                    ]:
                        if len(side_pts) < 3:
                            continue
                        vals = []
                        for px, py in side_pts:
                            if 0 <= py < h and 0 <= px < w:
                                if binary[py, px]:
                                    vals.append(float(gray_image[py, px]))
                        if len(vals) < 3:
                            continue
                        side_med = float(np.median(vals))

                        # Check if this side matches an unused cluster
                        # STRONGLY (diff > 15) — avoid minor corrections
                        best_unused = min(
                            unused_clusters,
                            key=lambda c: abs(c - side_med),
                        )
                        dist_to_unused = abs(best_unused - side_med)
                        dist_to_current = abs(current_centre - side_med)

                        # Only reassign if unused is much closer AND current was contaminated
                        if (dist_to_unused < 15 and 
                            dist_to_unused < dist_to_current and 
                            current_iqr > 30):  # only for noisy curves
                            old_centre = current_centre
                            signatures[cid] = (best_unused, 15.0)
                            # Update tracking
                            unused_clusters.remove(best_unused)
                            unused_clusters.append(old_centre)
                            logger.debug(
                                "  Unused-cluster fix: Curve %d %s-side=%.0f "
                                "-> cluster %.0f (was %.0f)",
                                cid, side_name, side_med,
                                best_unused, old_centre,
                            )
                            break  # fixed this curve

            shade_results = _shade_guided_reextract(
                gray_image, binary,
                corrected_curves,
                cluster_centres,
                signatures,
                x0, x1,
            )

            shade_worked = False
            for cid, seg in shade_results.items():
                if seg is not None and cid in corrected_curves:
                    # The segment may extend beyond [x0, x1] due to
                    # post-zone extension.  Stitch using the actual
                    # segment x-extent so we don't duplicate points.
                    seg_xs = [p[0] for p in seg]
                    seg_x0 = min(seg_xs)
                    seg_x1 = max(seg_xs)
                    corrected_curves[cid] = _stitch_segment(
                        corrected_curves[cid], seg, seg_x0, seg_x1,
                    )
                    shade_worked = True
                    logger.debug(
                        "  Curve %d: shade re-extracted (%d pts) in [%d,%d]",
                        cid, len(seg), seg_x0, seg_x1,
                    )

            if shade_worked:
                continue  # done with this zone — move to next

            logger.debug(
                "  Shade re-extraction failed; falling back to DP re-trace",
            )

        # ── FALLBACK: chunked DP re-tracing ──────────────────────────
        # Used when shade differences are too small (<8) or shade
        # re-extraction produced too few points.
        CHUNK_WIDTH = 80
        zone_width = x1 - x0

        if zone_width > CHUNK_WIDTH * 1.5:
            n_chunks = max(2, (zone_width + CHUNK_WIDTH - 1) // CHUNK_WIDTH)
            chunk_w = zone_width // n_chunks
            chunk_boundaries = [
                (x0 + i * chunk_w, x0 + (i + 1) * chunk_w)
                for i in range(n_chunks)
            ]
            chunk_boundaries[-1] = (chunk_boundaries[-1][0], x1)
            logger.debug(
                "  DP fallback: wide zone (%d px) → %d chunks",
                zone_width, n_chunks,
            )
        else:
            chunk_boundaries = [(x0, x1)]

        for chunk_x0, chunk_x1 in chunk_boundaries:
            chunk_y0, chunk_y1 = y0, y1

            for cid in cids:
                if cid not in corrected_curves or cid not in signatures:
                    continue

                mean_i, std_i = signatures[cid]

                other_sigs = [
                    signatures[oid]
                    for oid in cids
                    if oid != cid and oid in signatures
                ]

                new_segment = _retrace_through_zone(
                    gray_image,
                    enhanced_full,
                    binary,
                    corrected_curves[cid],
                    mean_i,
                    std_i,
                    chunk_x0, chunk_x1, chunk_y0, chunk_y1,
                    max_jump=max_jump,
                    intensity_weight=zone_int_w,
                    geometry_weight=zone_geo_w,
                    curvature_weight=curvature_weight,
                    approach_pts=max(15, (chunk_x1 - chunk_x0) // 3),
                    other_signatures=other_sigs if other_sigs else None,
                )

                if new_segment is None:
                    wider = max_jump + 4
                    new_segment = _retrace_through_zone(
                        gray_image,
                        enhanced_full,
                        binary,
                        corrected_curves[cid],
                        mean_i,
                        std_i,
                        chunk_x0, chunk_x1, chunk_y0, chunk_y1,
                        max_jump=wider,
                        intensity_weight=zone_int_w,
                        geometry_weight=zone_geo_w,
                        curvature_weight=curvature_weight,
                        approach_pts=max(15, (chunk_x1 - chunk_x0) // 3),
                        other_signatures=other_sigs if other_sigs else None,
                    )

                if new_segment is None:
                    logger.debug(
                        "  Curve %d: chunk [%d,%d] DP re-trace failed",
                        cid, chunk_x0, chunk_x1,
                    )
                    continue

                corrected_curves[cid] = _stitch_segment(
                    corrected_curves[cid], new_segment, chunk_x0, chunk_x1,
                )
                logger.debug(
                    "  Curve %d: DP chunk [%d,%d] replaced (%d pts)",
                    cid, chunk_x0, chunk_x1, len(new_segment),
                )

    # 4. Optional debug visualisation
    if debug_dir:
        _save_debug_viz(
            gray_image, enhanced_full, binary,
            curves, corrected_curves, zones, debug_dir,
        )

    return corrected_curves


# ====================================================================
# 6.  Stitch a corrected segment back into a curve
# ====================================================================

def _stitch_segment(
    full_path: List[Tuple[int, int]],
    new_segment: List[Tuple[int, int]],
    zone_x_start: int,
    zone_x_end: int,
) -> List[Tuple[int, int]]:
    """Replace the portion of *full_path* inside [zone_x_start, zone_x_end]
    with *new_segment*, preserving points outside the zone."""
    # Keep points before the zone
    left = [(px, py) for px, py in full_path if px < zone_x_start]
    # Keep points after the zone
    right = [(px, py) for px, py in full_path if px > zone_x_end]

    # Sort segments by x
    left.sort(key=lambda p: p[0])
    new_segment_sorted = sorted(new_segment, key=lambda p: p[0])
    right.sort(key=lambda p: p[0])

    result = left + new_segment_sorted + right
    return result


# ====================================================================
# 7.  Debug visualisation
# ====================================================================

def _save_debug_viz(
    gray: np.ndarray,
    enhanced: np.ndarray,
    binary: np.ndarray,
    original_curves: Dict[int, List[Tuple[int, int]]],
    corrected_curves: Dict[int, List[Tuple[int, int]]],
    zones: List[Dict[str, Any]],
    debug_dir: str,
) -> None:
    """Save debug images showing before/after intersection correction."""
    try:
        from pathlib import Path
        from PIL import Image, ImageDraw

        out = Path(debug_dir)
        out.mkdir(parents=True, exist_ok=True)

        h, w = gray.shape

        colors = [
            (255, 0, 0),      # red
            (0, 200, 0),      # green
            (0, 80, 255),     # blue
            (255, 165, 0),    # orange
            (180, 0, 255),    # purple
            (255, 255, 0),    # yellow
        ]

        # --- Before overlay ---
        base_rgb = np.stack([gray, gray, gray], axis=2)
        overlay = Image.fromarray(base_rgb)
        draw = ImageDraw.Draw(overlay)

        for idx, pts in original_curves.items():
            c = colors[idx % len(colors)]
            for i in range(len(pts) - 1):
                draw.line([pts[i], pts[i + 1]], fill=c, width=2)

        # Draw zone rectangles
        for zone in zones:
            x0, x1 = zone["x_start"], zone["x_end"]
            y0, y1 = _zone_y_bounds(original_curves, zone, h)
            draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 0), width=2)

        overlay.save(str(out / "intersection_before.png"))

        # --- After overlay ---
        overlay2 = Image.fromarray(base_rgb)
        draw2 = ImageDraw.Draw(overlay2)

        for idx, pts in corrected_curves.items():
            c = colors[idx % len(colors)]
            for i in range(len(pts) - 1):
                draw2.line([pts[i], pts[i + 1]], fill=c, width=2)

        for zone in zones:
            x0, x1 = zone["x_start"], zone["x_end"]
            y0, y1 = _zone_y_bounds(corrected_curves, zone, h)
            draw2.rectangle([x0, y0, x1, y1], outline=(0, 255, 255), width=2)

        overlay2.save(str(out / "intersection_after.png"))

        # --- Enhanced ROI for each zone ---
        enhanced_u8 = (np.clip(enhanced, 0, 1) * 255).astype(np.uint8)
        for zi, zone in enumerate(zones):
            x0, x1 = zone["x_start"], zone["x_end"]
            y0, y1 = _zone_y_bounds(original_curves, zone, h)
            x0c = max(0, x0)
            x1c = min(w, x1 + 1)
            y0c = max(0, y0)
            y1c = min(h, y1 + 1)
            roi_enh = enhanced_u8[y0c:y1c, x0c:x1c]
            roi_orig = gray[y0c:y1c, x0c:x1c]
            Image.fromarray(roi_orig).save(
                str(out / f"intersection_zone{zi}_original.png")
            )
            Image.fromarray(roi_enh).save(
                str(out / f"intersection_zone{zi}_enhanced.png")
            )

        logger.info("Intersection debug images saved to %s", debug_dir)

    except Exception as exc:
        logger.debug("_save_debug_viz failed: %s", exc)
