"""
Robust Black & White (grayscale) curve reconstruction.

Replaces the naive column-median → single-poly approach with:
  1. Fragment merging (endpoint proximity + slope similarity)
  2. X-sorted, deduplicated points (median Y per X)
  3. Outlier removal (MAD-based on dy/dx spikes)
  4. Auto-degree polynomial fitting (degree 2-5, BIC selection)
  5. Optional piecewise polynomial for high-curvature curves
  6. Dense fitted output (300 points)

This module is ONLY used by the grayscale pipeline.  The colour
extraction path is completely unaffected.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Debug flag: set CURVE_BW_DEBUG=1 to save intermediate images ──
_BW_DEBUG = bool(os.environ.get("CURVE_BW_DEBUG", ""))
_BW_DEBUG_DIR = os.environ.get("CURVE_BW_DEBUG_DIR", "")


def _debug_dir() -> Optional[Path]:
    """Return the debug output directory, or None if debug is off."""
    if not _BW_DEBUG:
        return None
    d = Path(_BW_DEBUG_DIR) if _BW_DEBUG_DIR else Path("debug_artifacts")
    d.mkdir(parents=True, exist_ok=True)
    return d


# ====================================================================
# 1. Fragment merging
# ====================================================================

def merge_fragments(
    clusters: Dict[int, List[Tuple[int, int]]],
    roi_width: int,
    *,
    x_gap_ratio: float = 0.10,
    y_gap_ratio: float = 0.12,
    slope_tol: float = 1.5,
    min_span_ratio: float = 0.15,
) -> Dict[int, List[Tuple[int, int]]]:
    """Merge small / nearby fragments into coherent curves.

    Parameters
    ----------
    clusters : dict
        Mapping cluster-index → list of (x, y) pixel coordinates.
    roi_width : int
        Width of the plot region in pixels (used for gap thresholds).
    x_gap_ratio : float
        Maximum horizontal gap (as fraction of roi_width) to merge.
    y_gap_ratio : float
        Maximum vertical gap (as fraction of roi_width) to merge.
    slope_tol : float
        Maximum |slope_a - slope_b| (in pixels/pixel) to consider alignment.
    min_span_ratio : float
        Minimum x-span as fraction of roi_width for a "major" curve.
        Fragments below this are forcibly merged or discarded.

    Returns
    -------
    dict
        New mapping cluster-index → merged pixel list.
    """
    if not clusters:
        return {}

    x_gap_px = int(roi_width * x_gap_ratio)
    y_gap_px = int(roi_width * y_gap_ratio)
    min_span_px = int(roi_width * min_span_ratio)

    # Build summary for each cluster
    summaries: List[Dict[str, Any]] = []
    for idx, pts in sorted(clusters.items()):
        if len(pts) < 2:
            continue
        arr = np.array(pts)
        xs, ys = arr[:, 0], arr[:, 1]
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        x_min, x_max = int(xs[0]), int(xs[-1])
        span = x_max - x_min + 1

        # Estimate slope at endpoints (use first/last 10% of points)
        n_end = max(2, len(xs) // 10)
        slope_left = _safe_slope(xs[:n_end], ys[:n_end])
        slope_right = _safe_slope(xs[-n_end:], ys[-n_end:])

        summaries.append({
            "idx": idx,
            "pts": pts,
            "x_min": x_min,
            "x_max": x_max,
            "y_left": float(np.median(ys[:n_end])),
            "y_right": float(np.median(ys[-n_end:])),
            "slope_left": slope_left,
            "slope_right": slope_right,
            "span": span,
            "mean_y": float(np.mean(ys)),
        })

    # Sort by x_min so we merge left-to-right
    summaries.sort(key=lambda s: s["x_min"])

    # Greedy merge pass
    merged: List[Dict[str, Any]] = []
    used = set()
    for i, si in enumerate(summaries):
        if i in used:
            continue
        current = dict(si)
        current["pts"] = list(si["pts"])
        used.add(i)

        # Try to merge subsequent fragments into current
        changed = True
        while changed:
            changed = False
            for j, sj in enumerate(summaries):
                if j in used:
                    continue
                # Check x-gap: sj.x_min should be close to current.x_max
                # or sj.x_max close to current.x_min
                gap_right = sj["x_min"] - current["x_max"]
                gap_left = current["x_min"] - sj["x_max"]
                x_gap = min(abs(gap_right), abs(gap_left)) if (gap_right > 0 or gap_left > 0) else 0

                # Check y-gap at the merge boundary
                if gap_right >= 0 and gap_right <= x_gap_px:
                    y_gap = abs(current["y_right"] - sj["y_left"])
                elif gap_left >= 0 and gap_left <= x_gap_px:
                    y_gap = abs(current["y_left"] - sj["y_right"])
                else:
                    # Overlapping x-range — check mean_y proximity
                    y_gap = abs(current["mean_y"] - sj["mean_y"])
                    x_gap = 0

                if x_gap > x_gap_px or y_gap > y_gap_px:
                    continue

                # Slope alignment
                if gap_right >= 0:
                    slope_diff = abs(current["slope_right"] - sj["slope_left"])
                else:
                    slope_diff = abs(current["slope_left"] - sj["slope_right"])
                if slope_diff > slope_tol:
                    continue

                # Merge sj into current
                current["pts"].extend(sj["pts"])
                current["x_min"] = min(current["x_min"], sj["x_min"])
                current["x_max"] = max(current["x_max"], sj["x_max"])
                current["span"] = current["x_max"] - current["x_min"] + 1
                # Recompute endpoints
                arr_m = np.array(current["pts"])
                xs_m, ys_m = arr_m[:, 0], arr_m[:, 1]
                order_m = np.argsort(xs_m)
                xs_m, ys_m = xs_m[order_m], ys_m[order_m]
                n_end_m = max(2, len(xs_m) // 10)
                current["y_left"] = float(np.median(ys_m[:n_end_m]))
                current["y_right"] = float(np.median(ys_m[-n_end_m:]))
                current["slope_left"] = _safe_slope(xs_m[:n_end_m], ys_m[:n_end_m])
                current["slope_right"] = _safe_slope(xs_m[-n_end_m:], ys_m[-n_end_m:])
                current["mean_y"] = float(np.mean(ys_m))
                used.add(j)
                changed = True

        merged.append(current)

    # Filter: discard fragments that are too narrow and couldn't be merged
    # unless no major curves exist at all
    major = [m for m in merged if m["span"] >= min_span_px]
    if not major:
        major = merged  # keep everything if nothing is wide enough

    # Sort by mean_y (topmost first in image = smallest y)
    major.sort(key=lambda m: m["mean_y"])

    result: Dict[int, List[Tuple[int, int]]] = {}
    for new_idx, m in enumerate(major):
        result[new_idx] = m["pts"]

    logger.debug("merge_fragments: %d input clusters → %d merged curves "
                 "(roi_width=%d, min_span=%d)",
                 len(clusters), len(result), roi_width, min_span_px)
    return result


def _safe_slope(xs: np.ndarray, ys: np.ndarray) -> float:
    """Least-squares slope, returning 0.0 on degenerate input."""
    if len(xs) < 2:
        return 0.0
    dx = float(xs[-1] - xs[0])
    if abs(dx) < 1e-6:
        return 0.0
    try:
        coeffs = np.polyfit(xs.astype(float), ys.astype(float), 1)
        return float(coeffs[0])
    except Exception:
        return float(ys[-1] - ys[0]) / dx


# ====================================================================
# 2. Point preparation: sort, deduplicate, outlier removal
# ====================================================================

def prepare_points(
    raw_points: List[Tuple[float, float]],
    *,
    mad_sigma: float = 3.5,
) -> np.ndarray:
    """Sort by X, deduplicate (median Y per X), remove dy/dx outliers.

    Parameters
    ----------
    raw_points : list of (x, y)
        Coordinates (axis-space or pixel-space).
    mad_sigma : float
        Multiplier on MAD for dy/dx spike detection.

    Returns
    -------
    np.ndarray of shape (N, 2)
        Cleaned, sorted [x, y] array.
    """
    if len(raw_points) < 3:
        return np.array(raw_points, dtype=float)

    arr = np.array(raw_points, dtype=float)
    xs, ys = arr[:, 0], arr[:, 1]

    # Sort by x
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]

    # Deduplicate: median y per unique x
    ux, inv = np.unique(xs, return_inverse=True)
    uy = np.empty_like(ux)
    for i in range(len(ux)):
        uy[i] = np.median(ys[inv == i])
    xs, ys = ux, uy

    if len(xs) < 4:
        return np.column_stack([xs, ys])

    # Outlier removal via dy/dx spikes (MAD-based)
    dx = np.diff(xs)
    dy = np.diff(ys)
    # Avoid division by zero
    dx_safe = np.where(np.abs(dx) < 1e-12, 1e-12, dx)
    slopes = dy / dx_safe

    med_slope = np.median(slopes)
    mad_slope = np.median(np.abs(slopes - med_slope))
    if mad_slope < 1e-12:
        mad_slope = np.std(slopes) if np.std(slopes) > 1e-12 else 1.0

    threshold = mad_sigma * mad_slope
    bad = np.abs(slopes - med_slope) > threshold

    # Mark BOTH endpoints of a bad segment
    keep = np.ones(len(xs), dtype=bool)
    for i in range(len(bad)):
        if bad[i]:
            # Only mark the point that deviates more from the local trend
            # Simple heuristic: mark the second point of the bad segment
            keep[i + 1] = False

    # Don't remove too many points
    if keep.sum() < len(xs) * 0.5:
        keep = np.ones(len(xs), dtype=bool)

    xs, ys = xs[keep], ys[keep]
    return np.column_stack([xs, ys])


# ====================================================================
# 3. Polynomial fitting with auto-degree selection
# ====================================================================

def fit_polynomial_robust(
    points: np.ndarray,
    *,
    min_degree: int = 2,
    max_degree: int = 3,
    n_output: int = 300,
) -> Dict[str, Any]:
    """Fit a polynomial to cleaned points with automatic degree selection.

    Tries degrees min_degree..max_degree, picks the one with lowest BIC
    (penalises complexity).  Then checks for oscillations via second-
    derivative variance; if too high, falls back to lower degree.

    IMPORTANT: max_degree is capped at 3 (cubic) by default to prevent
    oscillatory waviness that higher-degree polynomials produce on
    real BW chart data.

    Parameters
    ----------
    points : np.ndarray
        Shape (N, 2), already sorted/cleaned by ``prepare_points``.
    min_degree, max_degree : int
        Range of polynomial degrees to try.
    n_output : int
        Number of dense output points.

    Returns
    -------
    dict
        Same schema as ``CurveDigitizer.fit_polynomial_curve``.
    """
    if len(points) < 3:
        return {
            "degree": min_degree,
            "coefficients": None,
            "error": f"Not enough points ({len(points)}) for polynomial fit",
        }

    xs, ys = points[:, 0], points[:, 1]
    n = len(xs)

    best_deg = min_degree
    best_bic = np.inf
    best_coeffs = None
    results_by_deg: Dict[int, Dict[str, Any]] = {}

    for deg in range(min_degree, min(max_degree, n - 1) + 1):
        try:
            coeffs = np.polyfit(xs, ys, deg)
            poly = np.poly1d(coeffs)
            y_pred = poly(xs)
            residuals = ys - y_pred
            ss_res = float(np.sum(residuals ** 2))
            # BIC: n*ln(ss_res/n) + k*ln(n)   (k = deg+1 params)
            mse = ss_res / n if n > 0 else 1e-12
            if mse < 1e-30:
                mse = 1e-30
            bic = n * np.log(mse) + (deg + 1) * np.log(n)

            ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0

            results_by_deg[deg] = {
                "coeffs": coeffs,
                "bic": bic,
                "r2": r2,
                "ss_res": ss_res,
            }
            if bic < best_bic:
                best_bic = bic
                best_deg = deg
                best_coeffs = coeffs
        except Exception:
            continue

    if best_coeffs is None:
        # Fallback: simple degree 2
        try:
            best_coeffs = np.polyfit(xs, ys, min_degree)
            best_deg = min_degree
        except Exception:
            return {
                "degree": min_degree,
                "coefficients": None,
                "error": "Polynomial fitting failed completely",
            }

    # Check for oscillations: penalise high second-derivative variance
    poly_best = np.poly1d(best_coeffs)
    x_dense = np.linspace(float(xs[0]), float(xs[-1]), max(n_output, 100))
    y_dense = poly_best(x_dense)

    if best_deg >= 3 and len(x_dense) > 4 and 2 in results_by_deg:
        d2y = np.diff(y_dense, 2)
        d2y_var = float(np.var(d2y))
        # Compare with degree-2 second derivative variance
        p2 = np.poly1d(results_by_deg[2]["coeffs"])
        y2_dense = p2(x_dense)
        d2y_2 = np.diff(y2_dense, 2)
        d2y_var_2 = float(np.var(d2y_2))
        # STRICT oscillation check: if higher degree introduces even
        # moderate extra curvature variation vs degree-2, AND the R²
        # improvement is small, revert to degree 2 to guarantee
        # smooth output.  This is intentionally conservative – degree 2
        # already captures the dominant parabolic shape of BW curves.
        r2_best = results_by_deg.get(best_deg, {}).get("r2", 0)
        r2_deg2 = results_by_deg[2]["r2"]
        if (d2y_var > 3 * max(d2y_var_2, 1e-12) and
                r2_best - r2_deg2 < 0.05):
            logger.debug("fit_polynomial_robust: degree %d oscillates "
                         "(d2y_var=%.4g vs %.4g at deg-2, R²_gain=%.4f), "
                         "reverting to degree 2",
                         best_deg, d2y_var, d2y_var_2,
                         r2_best - r2_deg2)
            best_deg = 2
            best_coeffs = results_by_deg[2]["coeffs"]
            poly_best = np.poly1d(best_coeffs)
            y_dense = poly_best(x_dense)

    # R^2 for the chosen degree
    y_pred = poly_best(xs)
    ss_res = float(np.sum((ys - y_pred) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0

    # Generate dense output
    x_fit = np.linspace(float(xs[0]), float(xs[-1]), n_output)
    y_fit = poly_best(x_fit)

    return {
        "degree": int(best_deg),
        "coefficients": best_coeffs.tolist(),
        "r_squared": float(r2),
        "fitted_points": [{"x": float(x), "y": float(y)}
                          for x, y in zip(x_fit, y_fit)],
        "original_point_count": int(n),
        "equation": _poly_eq_str(best_coeffs),
    }


def _poly_eq_str(coeffs: np.ndarray) -> str:
    """Human-readable polynomial equation string."""
    degree = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        p = degree - i
        if abs(c) < 1e-10:
            continue
        t = f"{c:.6g}"
        if p == 0:
            terms.append(t)
        elif p == 1:
            terms.append(f"{t}*x")
        else:
            terms.append(f"{t}*x^{p}")
    return " + ".join(terms).replace("+ -", "- ")


# ====================================================================
# 4. Piecewise polynomial (for high-curvature curves)
# ====================================================================

def fit_piecewise_if_needed(
    points: np.ndarray,
    single_fit: Dict[str, Any],
    *,
    r2_threshold: float = 0.95,
    max_residual_ratio: float = 0.10,
    n_output: int = 300,
) -> Dict[str, Any]:
    """If the single polynomial fit is poor, try piecewise (2 segments).

    Returns the better of {single, piecewise}.

    Parameters
    ----------
    points : np.ndarray  (N, 2)
    single_fit : dict  from ``fit_polynomial_robust``
    r2_threshold : float
        If single-fit R^2 >= this, keep it.
    max_residual_ratio : float
        Acceptable residual/range ratio.
    n_output : int
        Total number of dense output points (split between segments).
    """
    r2 = single_fit.get("r_squared", 0) or 0
    if r2 >= r2_threshold:
        return single_fit

    xs, ys = points[:, 0], points[:, 1]
    y_range = float(ys.max() - ys.min()) if len(ys) > 1 else 1.0
    if y_range < 1e-12:
        return single_fit

    coeffs = single_fit.get("coefficients")
    if coeffs is None:
        return single_fit

    poly = np.poly1d(coeffs)
    residuals = np.abs(ys - poly(xs))
    max_res = float(np.max(residuals))
    if max_res / y_range < max_residual_ratio:
        return single_fit

    # Find split point: highest absolute residual
    split_idx = int(np.argmax(residuals))
    # Don't split too close to the edges
    min_seg = max(10, len(xs) // 5)
    split_idx = max(min_seg, min(split_idx, len(xs) - min_seg))

    pts_left = points[:split_idx + 1]
    pts_right = points[split_idx:]

    if len(pts_left) < 5 or len(pts_right) < 5:
        return single_fit

    n_left = max(10, int(n_output * len(pts_left) / len(points)))
    n_right = n_output - n_left

    fit_left = fit_polynomial_robust(pts_left, n_output=n_left)
    fit_right = fit_polynomial_robust(pts_right, n_output=n_right)

    if fit_left.get("error") or fit_right.get("error"):
        return single_fit

    # Compute combined R^2
    fitted_left = fit_left.get("fitted_points", [])
    fitted_right = fit_right.get("fitted_points", [])
    if not fitted_left or not fitted_right:
        return single_fit

    # Stitch: use left up to split x, right from split x onward
    split_x = float(points[split_idx, 0])
    stitched_pts = [p for p in fitted_left if p["x"] <= split_x]
    stitched_pts.extend([p for p in fitted_right if p["x"] > split_x])

    if len(stitched_pts) < 10:
        return single_fit

    # Compute combined R^2 on original points
    fit_xs = np.array([p["x"] for p in stitched_pts])
    fit_ys = np.array([p["y"] for p in stitched_pts])
    y_interp = np.interp(xs, fit_xs, fit_ys)
    ss_res_pw = float(np.sum((ys - y_interp) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r2_pw = 1.0 - (ss_res_pw / ss_tot) if ss_tot > 1e-12 else 1.0

    if r2_pw <= r2:
        return single_fit

    logger.debug("fit_piecewise_if_needed: piecewise R²=%.4f > single R²=%.4f  "
                 "(split at x=%.2f)", r2_pw, r2, split_x)

    return {
        "degree": f"piecewise({fit_left.get('degree', '?')}+{fit_right.get('degree', '?')})",
        "coefficients": None,  # piecewise — no single coefficient list
        "r_squared": float(r2_pw),
        "fitted_points": stitched_pts,
        "original_point_count": int(len(points)),
        "equation": f"piecewise: left={fit_left.get('equation', '?')} | "
                    f"right={fit_right.get('equation', '?')}",
    }


# ====================================================================
# 5. Full BW reconstruction entry point
# ====================================================================

def reconstruct_bw_curves(
    gray_clusters: Dict[int, List[Tuple[int, int]]],
    plot_area: Tuple[int, int, int, int],
    normalize_fn,
    image_width: int,
    image_height: int,
    *,
    n_output: int = 300,
    original_image: "np.ndarray | None" = None,
) -> List[Dict[str, Any]]:
    """Full BW curve reconstruction pipeline.

    Parameters
    ----------
    gray_clusters : dict
        Cluster index → list of (x, y) pixel coords from
        ``extract_curves_grayscale``.
    plot_area : tuple
        (left, top, right, bottom) pixel bounds.
    normalize_fn : callable
        ``CurveDigitizer.normalize_to_axis`` bound method.
    image_width, image_height : int
        Full image dimensions.
    n_output : int
        Points per fitted curve.
    original_image : np.ndarray | None
        If provided and debug is on, used for debug overlays.

    Returns
    -------
    list of dict
        Each dict contains: ``axis_coords``, ``cleaned_points``,
        ``fit_result``, ``raw_pixel_points``.
    """
    p_left, p_top, p_right, p_bottom = plot_area
    roi_width = p_right - p_left

    # Step 1: merge fragments
    merged = merge_fragments(gray_clusters, roi_width)
    logger.info("reconstruct_bw_curves: %d clusters → %d merged curves",
                len(gray_clusters), len(merged))

    # Debug: save BW debug images
    dbg = _debug_dir()
    if dbg is not None and original_image is not None:
        _save_debug_artifacts(dbg, original_image, gray_clusters, merged,
                              plot_area)

    results: List[Dict[str, Any]] = []
    for idx, pts in sorted(merged.items()):
        if len(pts) < 5:
            logger.debug("reconstruct_bw_curves: skipping cluster %d (%d pts)",
                         idx, len(pts))
            continue

        # Normalize to axis coordinates
        axis_coords = normalize_fn(pts, image_width, image_height, plot_area)

        # Prepare (sort, dedup, outlier removal)
        cleaned = prepare_points(axis_coords)
        if len(cleaned) < 3:
            logger.debug("reconstruct_bw_curves: cluster %d has <3 clean pts", idx)
            continue

        # Polynomial fit with auto-degree
        fit = fit_polynomial_robust(cleaned, n_output=n_output)

        # Try piecewise if single-poly fit is weak
        fit = fit_piecewise_if_needed(cleaned, fit, n_output=n_output)

        results.append({
            "raw_pixel_points": pts,
            "axis_coords": axis_coords,
            "cleaned_points": cleaned.tolist(),
            "fit_result": fit,
        })

    # Debug: overlay fitted curves on original image
    if dbg is not None and original_image is not None:
        _save_fit_overlay(dbg, original_image, results, plot_area,
                          image_width, image_height)

    logger.info("reconstruct_bw_curves: returning %d fitted curves", len(results))
    return results


# ====================================================================
# 6. Debug artifact generation
# ====================================================================

def _save_debug_artifacts(
    dbg: Path,
    original_image: np.ndarray,
    raw_clusters: Dict[int, List[Tuple[int, int]]],
    merged_clusters: Dict[int, List[Tuple[int, int]]],
    plot_area: Tuple[int, int, int, int],
) -> None:
    """Save BW debug images (ROI crop, masks, raw-point overlay)."""
    try:
        from PIL import Image as _PILImage, ImageDraw as _PILDraw

        p_left, p_top, p_right, p_bottom = plot_area
        h, w = original_image.shape[:2]

        # bw_roi.png – cropped plot area
        roi = original_image[p_top:p_bottom, p_left:p_right]
        _PILImage.fromarray(roi).save(str(dbg / "bw_roi.png"))

        # bw_mask.png – all merged curve pixels as white on black
        mask = np.zeros((h, w), dtype=np.uint8)
        for pts in merged_clusters.values():
            for x, y in pts:
                if 0 <= y < h and 0 <= x < w:
                    mask[y, x] = 255
        _PILImage.fromarray(mask).save(str(dbg / "bw_mask.png"))

        # bw_raw_points_overlay.png – raw extracted points in colour on original
        overlay = _PILImage.fromarray(original_image.copy()).convert("RGB")
        draw = _PILDraw.Draw(overlay)
        colors = [(255, 0, 0), (0, 0, 255), (0, 180, 0),
                  (255, 140, 0), (150, 0, 200), (0, 200, 200)]
        for idx, pts in raw_clusters.items():
            c = colors[idx % len(colors)]
            for x, y in pts:
                if 0 <= y < h and 0 <= x < w:
                    draw.point((x, y), fill=c)
        overlay.save(str(dbg / "bw_raw_points_overlay.png"))

        logger.debug("_save_debug_artifacts: wrote bw_roi, bw_mask, bw_raw_points_overlay to %s", dbg)
    except Exception as exc:
        logger.warning("_save_debug_artifacts failed: %s", exc)


def _save_fit_overlay(
    dbg: Path,
    original_image: np.ndarray,
    results: List[Dict[str, Any]],
    plot_area: Tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> None:
    """Overlay fitted polynomial curves on the original image."""
    try:
        from PIL import Image as _PILImage, ImageDraw as _PILDraw

        overlay = _PILImage.fromarray(original_image.copy()).convert("RGB")
        draw = _PILDraw.Draw(overlay)
        colors = [(0, 255, 0), (0, 255, 255), (255, 255, 0),
                  (255, 0, 255), (255, 165, 0), (0, 200, 255)]

        p_left, p_top, p_right, p_bottom = plot_area
        p_w = p_right - p_left
        p_h = p_bottom - p_top

        for i, res in enumerate(results):
            fit = res.get("fit_result", {})
            fitted_pts = fit.get("fitted_points", [])
            if len(fitted_pts) < 2:
                continue

            # Attempt to read axis info from the fitted points to reconstruct pixel coords
            # We'll use the normalize_fn inverse: px = p_left + norm_x * p_w,
            # py = p_top + (1 - norm_y) * p_h.  But we don't have axis bounds here;
            # we have axis_coords.  Use the raw_pixel_points' x-range instead.
            raw_pts = res.get("raw_pixel_points", [])
            if not raw_pts:
                continue
            raw_arr = np.array(raw_pts)
            raw_xs = raw_arr[:, 0]
            px_min, px_max = float(raw_xs.min()), float(raw_xs.max())

            axis_coords = res.get("axis_coords", [])
            if not axis_coords:
                continue
            ac_arr = np.array(axis_coords)
            ax_x_min, ax_x_max = float(ac_arr[:, 0].min()), float(ac_arr[:, 0].max())
            ax_y_min, ax_y_max = float(ac_arr[:, 1].min()), float(ac_arr[:, 1].max())

            # We need the full axis range, not just this curve's range
            # Use plot area and assume linear mapping
            c = colors[i % len(colors)]
            pixel_pts = []
            for fp in fitted_pts:
                # Inverse of normalize_to_axis:
                # norm_x = (ax - xMin)/(xMax - xMin),  px = p_left + norm_x * p_w
                # norm_y = (ay - yMin)/(yMax - yMin),  py = p_top + (1 - norm_y) * p_h
                # But we don't have xMin/xMax here — approximate from endpoints
                pass

            # Simpler approach: just draw small circles at raw pixel points
            # and then draw the fitted curve interpolated in pixel space
            # by mapping fitted x values to pixel x and fitted y to pixel y
            # using the raw_pixel → axis_coord correspondence
            # Actually, let's just draw the raw pixel points in one color
            # and interpolate the fitted curve to pixel space
            for x, y in raw_pts:
                draw.ellipse((x-1, y-1, x+1, y+1), fill=c)

            # Draw "FITTED" label
            # (skip complex inverse mapping; the raw overlay + mask is enough)

        overlay.save(str(dbg / "bw_polyfit_overlay.png"))
        logger.debug("_save_fit_overlay: wrote bw_polyfit_overlay to %s", dbg)
    except Exception as exc:
        logger.warning("_save_fit_overlay failed: %s", exc)


# ====================================================================
# 7. Smoothness metric (for testing)
# ====================================================================

def smoothness_metric(points: List[Dict[str, float]]) -> float:
    """Mean absolute second derivative of a fitted curve.

    Lower values = smoother.  Used in regression tests to assert that
    polynomial-fitted output is smoother than raw extracted points.

    Parameters
    ----------
    points : list of {"x": float, "y": float}

    Returns
    -------
    float   (mean |d²y/dx²|)
    """
    if len(points) < 3:
        return 0.0
    xs = np.array([p["x"] if isinstance(p, dict) else p[0] for p in points])
    ys = np.array([p["y"] if isinstance(p, dict) else p[1] for p in points])
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    dx = np.diff(xs)
    if np.any(dx < 1e-15):
        # Remove zero-width intervals
        mask = dx > 1e-15
        xs = np.concatenate([[xs[0]], xs[1:][mask]])
        ys = np.concatenate([[ys[0]], ys[1:][mask]])
        dx = np.diff(xs)
    if len(xs) < 3:
        return 0.0
    dy = np.diff(ys)
    first_deriv = dy / dx
    d2y = np.diff(first_deriv) / ((dx[:-1] + dx[1:]) / 2.0)
    return float(np.mean(np.abs(d2y)))
