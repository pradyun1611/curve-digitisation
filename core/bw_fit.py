"""
Robust BW curve fitting — centerline extraction + polynomial fit.

Replaces fixed degree-2 polyfit for BW-extracted point clouds.
NO spline fitting — polynomial only (degree 2-4) with shape enforcement.

Pipeline
--------
1. **Centerline extraction** — bin by x, take median y per bin, discard thin bins.
2. **Outlier removal** — Hampel filter on binned y, then iterative sigma-clip.
3. **Pre-smooth** — light Savitzky-Golay on binned series before fitting.
4. **Fit** — BIC-selected polynomial (deg 2–4, prefer 2-3).
5. **Shape sanity** — enforce unimodal arc (≤1 derivative sign change),
   prefer concave-down; downgrade degree if violated.
6. **Sample** — 300 equispaced points over observed x-range.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ====================================================================
# 1. Centerline extraction  — pixel/axis cloud  ➜  single-valued y(x)
# ====================================================================

def extract_centerline(
    coords: List[Tuple[float, float]],
    *,
    n_bins: Optional[int] = None,
    min_points_per_bin: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a thick (x, y) point cloud into a clean centerline.

    Parameters
    ----------
    coords : list of (x, y)
        Raw axis-space coordinates (may have many y per x due to stroke width).
    n_bins : int or None
        Number of x-bins.  *None* → auto (clamp to 250-600 based on x-span).
    min_points_per_bin : int
        Bins with fewer points are discarded.

    Returns
    -------
    x_bin, y_med : 1-D arrays
        Sorted, single-valued centerline series.
    """
    arr = np.array(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.array([]), np.array([])

    xs, ys = arr[:, 0], arr[:, 1]
    x_min, x_max = xs.min(), xs.max()
    x_span = x_max - x_min
    if x_span < 1e-12:
        return np.array([x_min]), np.array([np.median(ys)])

    if n_bins is None:
        n_bins = int(np.clip(x_span * 3, 250, 600))
    n_bins = max(n_bins, 10)

    edges = np.linspace(x_min, x_max, n_bins + 1)
    # Assign each point to a bin
    bin_idx = np.searchsorted(edges[1:], xs, side="right")
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    x_out, y_out = [], []
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count < min_points_per_bin:
            continue
        bx = xs[mask]
        by = ys[mask]
        x_out.append(float(np.median(bx)))
        # Trimmed-mean-like: use median (robust to thick strokes)
        y_out.append(float(np.median(by)))

    x_out = np.array(x_out, dtype=np.float64)
    y_out = np.array(y_out, dtype=np.float64)
    order = np.argsort(x_out)
    return x_out[order], y_out[order]


# ====================================================================
# 2. Robust outlier removal
# ====================================================================

def _hampel_filter(
    y: np.ndarray, *, half_window: int = 5, n_sigma: float = 3.0,
) -> np.ndarray:
    """Hampel identifier — replace outlier y values with local median."""
    n = len(y)
    y_clean = y.copy()
    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        local = y[lo:hi]
        med = np.median(local)
        mad = np.median(np.abs(local - med))
        mad_scaled = 1.4826 * mad  # consistent estimator of σ
        if mad_scaled < 1e-12:
            mad_scaled = float(np.std(local))
        if mad_scaled < 1e-12:
            continue
        if abs(y[i] - med) > n_sigma * mad_scaled:
            y_clean[i] = med
    return y_clean


def _sigma_clip_on_residuals(
    x: np.ndarray, y: np.ndarray,
    *, degree: int = 3, sigma: float = 2.5, max_iter: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Iterative sigma-clip based on polynomial residuals."""
    xw, yw = x.copy(), y.copy()
    for _ in range(max_iter):
        if len(xw) < degree + 2:
            break
        try:
            coeffs = np.polyfit(xw, yw, degree)
        except Exception:
            break
        pred = np.polyval(coeffs, xw)
        res = np.abs(yw - pred)
        med_res = np.median(res)
        if med_res < 1e-12:
            break
        keep = res <= sigma * med_res
        if keep.all() or keep.sum() < len(x) * 0.5:
            break
        xw, yw = xw[keep], yw[keep]
    return xw, yw


def remove_outliers(
    x: np.ndarray, y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Combined Hampel + sigma-clip outlier removal on a binned series."""
    if len(x) < 6:
        return x, y
    y_clean = _hampel_filter(y, half_window=5, n_sigma=3.0)
    return _sigma_clip_on_residuals(x, y_clean, degree=3, sigma=2.5, max_iter=3)


# ====================================================================
# 3. Pre-smooth (Savitzky-Golay on binned series)
# ====================================================================

def presmooth(
    y: np.ndarray, *, window: int = 0,
) -> np.ndarray:
    """Light Savitzky-Golay pre-smooth.  window=0 → auto."""
    if len(y) < 7:
        return y
    if window <= 0:
        window = max(9, min(len(y) // 6 | 1, 31))
    if window % 2 == 0:
        window += 1
    window = min(window, len(y))
    if window % 2 == 0:
        window -= 1
    if window < 5:
        return y
    try:
        from scipy.signal import savgol_filter
        polyorder = 2 if window < 9 else 3
        return savgol_filter(y, window_length=window, polyorder=polyorder)
    except Exception:
        return y


# ====================================================================
# 4. Polynomial fit with BIC degree selection (deg 2-4 only)
# ====================================================================

def _bic_score(n: int, rss: float, k: int) -> float:
    """Bayesian Information Criterion (lower is better)."""
    if rss <= 0 or n <= k:
        return float("inf")
    return n * math.log(rss / n) + k * math.log(n)


def _poly_equation_string(coeffs: np.ndarray, deg: int) -> str:
    """Build a human-readable equation string."""
    terms = []
    for i, c in enumerate(coeffs):
        power = deg - i
        if abs(c) < 1e-15:
            continue
        if power == 0:
            terms.append(f"{c:.6g}")
        elif power == 1:
            terms.append(f"{c:.6g}*x")
        else:
            terms.append(f"{c:.6g}*x^{power}")
    return " + ".join(terms) if terms else "0"


def _fit_poly_bic(
    x: np.ndarray, y: np.ndarray,
    *, min_degree: int = 1, max_degree: int = 4, n_output: int = 300,
) -> Dict[str, Any]:
    """Fit polynomial, choose degree by BIC.  Max degree capped to 4.

    Starts from degree 1 (linear) so straight/slanted lines stay straight.
    Higher degrees are only chosen when they significantly improve BIC.
    """
    max_degree = min(max_degree, 4)  # hard cap
    min_degree = max(min_degree, 1)  # allow linear
    candidates_fit: Dict[int, Dict[str, Any]] = {}
    bic_by_deg: Dict[int, float] = {}
    best: Optional[Dict[str, Any]] = None
    best_bic = float("inf")
    n = len(x)

    for deg in range(min_degree, max_degree + 1):
        if n < deg + 1:
            continue
        try:
            coeffs = np.polyfit(x, y, deg)
        except Exception:
            continue
        pred = np.polyval(coeffs, x)
        rss = float(np.sum((y - pred) ** 2))
        bic = _bic_score(n, rss, deg + 1)
        if bic < best_bic:
            best_bic = bic

            x_fit = np.linspace(float(x[0]), float(x[-1]), n_output)
            y_fit = np.polyval(coeffs, x_fit)

            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - rss / ss_tot if ss_tot > 1e-12 else 1.0
            rmse = math.sqrt(rss / max(n, 1))

            best = {
                "degree": deg,
                "coefficients": coeffs.tolist(),
                "r_squared": float(r2),
                "fitted_points": [
                    {"x": float(xv), "y": float(yv)}
                    for xv, yv in zip(x_fit, y_fit)
                ],
                "original_point_count": n,
                "equation": _poly_equation_string(coeffs, deg),
                "fit_method": f"poly_{deg}",
                "fit_rmse_on_centerline": float(rmse),
            }
            candidates_fit[deg] = best
            bic_by_deg[deg] = bic
        else:
            candidates_fit[deg] = {
                "degree": deg,
                "coefficients": coeffs.tolist(),
                "r_squared": float(1.0 - rss / float(np.sum((y - np.mean(y)) ** 2)))
                if float(np.sum((y - np.mean(y)) ** 2)) > 1e-12 else 1.0,
                "fitted_points": [
                    {"x": float(xv), "y": float(yv)}
                    for xv, yv in zip(
                        np.linspace(float(x[0]), float(x[-1]), n_output),
                        np.polyval(coeffs, np.linspace(float(x[0]), float(x[-1]), n_output)),
                    )
                ],
                "original_point_count": n,
                "equation": _poly_equation_string(coeffs, deg),
                "fit_method": f"poly_{deg}",
                "fit_rmse_on_centerline": float(math.sqrt(rss / max(n, 1))),
            }
            bic_by_deg[deg] = bic

    if best is None:
        # absolute fallback — degree 2 (safe default for performance curves)
        deg_fb = min(max_degree, max(min_degree, 2))
        coeffs = np.polyfit(x, y, deg_fb)
        x_fit = np.linspace(float(x[0]), float(x[-1]), n_output)
        y_fit = np.polyval(coeffs, x_fit)
        rss_fb = float(np.sum((y - np.polyval(coeffs, x)) ** 2))
        ss_tot_fb = float(np.sum((y - np.mean(y)) ** 2))
        r2_fb = 1.0 - rss_fb / ss_tot_fb if ss_tot_fb > 1e-12 else 1.0
        best = {
            "degree": deg_fb,
            "coefficients": coeffs.tolist(),
            "r_squared": float(r2_fb),
            "fitted_points": [
                {"x": float(xv), "y": float(yv)}
                for xv, yv in zip(x_fit, y_fit)
            ],
            "original_point_count": n,
            "equation": f"fallback poly({deg_fb})",
            "fit_method": f"poly_{deg_fb}",
            "fit_rmse_on_centerline": float(math.sqrt(rss_fb / max(n, 1))),
        }
    # Degree preference to avoid needless wiggles:
    # STRONGLY prefer lower degrees unless higher degree improves BIC a lot.
    # This prevents straight lines from being bent into arcs.
    if best is not None and isinstance(best.get("degree"), int):
        best_deg = int(best["degree"])

        if best_deg == 4 and 3 in bic_by_deg and 3 in candidates_fit:
            if bic_by_deg[4] - bic_by_deg[3] > -6.0:
                best = candidates_fit[3]
        if isinstance(best.get("degree"), int) and int(best["degree"]) == 3:
            if 2 in bic_by_deg and 2 in candidates_fit:
                if bic_by_deg[3] - bic_by_deg[2] > -3.0:
                    best = candidates_fit[2]
        # Prefer degree 1 (linear) over degree 2 only when the data
        # is extremely well-explained by a straight line.  Be conservative:
        # short segments of genuinely curved lines can have high linear R².
        if isinstance(best.get("degree"), int) and int(best["degree"]) == 2:
            if 1 in bic_by_deg and 1 in candidates_fit:
                r2_lin = candidates_fit[1].get("r_squared", 0)
                bic_diff = bic_by_deg[2] - bic_by_deg[1]
                # Only prefer linear when R² is near-perfect AND BIC
                # does not strongly favour quadratic
                if r2_lin >= 0.999 and bic_diff > -3.0:
                    best = candidates_fit[1]
                    logger.debug("_fit_poly_bic: preferring linear (R²=%.4f)", r2_lin)

    return best


# ====================================================================
# 5. Shape sanity — unimodal arc enforcement
# ====================================================================

def _count_sign_changes(y: np.ndarray) -> int:
    """Count sign changes in first derivative of y."""
    dy = np.diff(y)
    signs = np.sign(dy)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


def _curvature_instability(y: np.ndarray) -> Tuple[int, float]:
    """Return (sign_changes_in_d2y, dominant_sign_ratio)."""
    d2 = np.diff(y, n=2)
    if len(d2) < 3:
        return 0, 1.0
    s = np.sign(d2)
    s = s[s != 0]
    if len(s) < 2:
        return 0, 1.0
    sc = int(np.sum(s[1:] != s[:-1]))
    dom = max(float(np.mean(s > 0)), float(np.mean(s < 0)))
    return sc, dom


def _shape_sanity(
    fit: Dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_output: int = 300,
    max_sign_changes: int = 1,
    max_d2_sign_changes: int = 0,
    min_d2_dominance: float = 0.85,
) -> Dict[str, Any]:
    """Enforce unimodal arc shape (only for degree >= 2).

    Rules (applied in order):
    a) degree 1 (linear) always passes — straight lines are valid shapes;
    b) derivative should change sign at most *max_sign_changes* times
       (one peak/valley is OK → efficiency hump);
    c) if violated, reduce degree and refit until the constraint holds
       or we reach degree 2 (preferred minimum for performance curves).
    """
    fps = fit.get("fitted_points", [])
    if len(fps) < 10:
        return fit

    # Linear fits always pass shape sanity — they cannot invent curvature
    current_deg = fit.get("degree", 4)
    if not isinstance(current_deg, int):
        current_deg = 4
    if current_deg <= 1:
        return fit

    y_fit = np.array([p["y"] for p in fps])
    sc = _count_sign_changes(y_fit)
    d2_sc, d2_dom = _curvature_instability(y_fit)

    if sc <= max_sign_changes and d2_sc <= max_d2_sign_changes and d2_dom >= min_d2_dominance:
        return fit

    logger.info(
        "bw_fit shape_sanity: dy_sc=%d (max %d), d2_sc=%d (max %d), d2_dom=%.2f (min %.2f) at deg %d — downgrading",
        sc, max_sign_changes, d2_sc, max_d2_sign_changes, d2_dom, min_d2_dominance, current_deg,
    )

    # Try progressively lower degrees down to 2 (preferred minimum
    # for performance curves — degree 1 is too aggressive for most use cases)
    for try_deg in range(min(current_deg - 1, 3), 1, -1):
        refit = _fit_poly_bic(x, y, min_degree=try_deg, max_degree=try_deg,
                              n_output=n_output)
        y2 = np.array([p["y"] for p in refit.get("fitted_points", [])])
        if len(y2) >= 10:
            sc2 = _count_sign_changes(y2)
            d2_sc2, d2_dom2 = _curvature_instability(y2)
            if sc2 <= max_sign_changes and d2_sc2 <= max_d2_sign_changes and d2_dom2 >= min_d2_dominance:
                logger.info("bw_fit shape_sanity: deg %d passes", try_deg)
                return refit

    # Fallback: degree 2 (quadratic — safe minimum for performance curves)
    logger.info("bw_fit shape_sanity: falling back to deg 2")
    return _fit_poly_bic(x, y, min_degree=2, max_degree=2, n_output=n_output)


# ====================================================================
# 6. Public entry point — polynomial-only BW fitting
# ====================================================================

def fit_bw_curve(
    axis_coords: List[Tuple[float, float]],
    *,
    n_output: int = 300,
    min_points_per_bin: int = 3,
) -> Tuple[Dict[str, Any], List[Tuple[float, float]]]:
    """Fit a BW-extracted point cloud using centerline + robust polynomial.

    Parameters
    ----------
    axis_coords : list of (x, y)
        Normalised axis-space coordinates (raw, thick cloud).
    n_output : int
        Number of fitted-points to return.
    min_points_per_bin : int
        Minimum points per x-bin for centerline extraction.

    Returns
    -------
    fit_result : dict
        Same schema as ``fit_polynomial_curve``:
        keys ``degree``, ``coefficients``, ``r_squared``, ``fitted_points``,
        ``original_point_count``, ``equation``, plus BW extras:
        ``fit_method``, ``fit_rmse_on_centerline``, ``num_bins_used``.
    cleaned_coords : list of (x, y)
        The centerline-extracted, outlier-removed series actually used for fitting.
        Suitable for metrics computation downstream.
    """
    if len(axis_coords) < 4:
        return {
            "degree": 2,
            "coefficients": None,
            "r_squared": 0.0,
            "fitted_points": [],
            "original_point_count": len(axis_coords),
            "equation": "insufficient data",
            "fit_method": "none",
            "fit_rmse_on_centerline": 0.0,
            "num_bins_used": 0,
        }, list(axis_coords)

    # Remember the FULL x-range of the original data (including anchor
    # endpoints) so the fitted output covers the entire curve span,
    # even if centerline extraction or outlier removal trims the edges.
    all_xs = np.array([p[0] for p in axis_coords], dtype=np.float64)
    orig_x_min = float(all_xs.min())
    orig_x_max = float(all_xs.max())

    # 1. Centerline
    x_bin, y_bin = extract_centerline(
        axis_coords, min_points_per_bin=min_points_per_bin,
    )
    if len(x_bin) < 4:
        # Not enough bins — fall back to simple dedup (median per unique x)
        arr = np.array(sorted(axis_coords, key=lambda p: p[0]))
        ux, inv = np.unique(arr[:, 0], return_inverse=True)
        uy = np.array([np.median(arr[inv == i, 1]) for i in range(len(ux))])
        x_bin, y_bin = ux, uy

    num_bins = len(x_bin)

    # 2. Outlier removal
    x_clean, y_clean = remove_outliers(x_bin, y_bin)

    # 3. Pre-smooth
    y_smooth = presmooth(y_clean)

    # 4. Polynomial fit (degree 1-4, BIC selection)
    # min_degree=1 allows BIC to detect truly straight lines, but the
    # degree preference logic in _fit_poly_bic is conservative and only
    # chooses deg 1 when R² >= 0.999.
    fit = _fit_poly_bic(x_clean, y_smooth, min_degree=1, max_degree=4,
                        n_output=n_output)

    # 5. Shape sanity — enforce unimodal arc
    fit = _shape_sanity(fit, x_clean, y_smooth, n_output=n_output,
                        max_d2_sign_changes=3, min_d2_dominance=0.70)

    # 5b. Extend fitted points to cover the full original x-range.
    #     Centerline extraction and outlier removal can shrink the
    #     x-range, causing curves to start too late or end too early.
    #     Re-evaluate the polynomial over the original range.
    #     GUARD: for extrapolated regions (outside the fitted x-range),
    #     use linear extension from the edge slope to prevent polynomial
    #     divergence (high-degree polys can blow up beyond data bounds).
    coeffs = fit.get("coefficients")
    if coeffs is not None and (orig_x_max - orig_x_min) > 1e-12:
        fit_x_min = fit["fitted_points"][0]["x"] if fit["fitted_points"] else orig_x_min
        fit_x_max = fit["fitted_points"][-1]["x"] if fit["fitted_points"] else orig_x_max
        if orig_x_min < fit_x_min - 1e-9 or orig_x_max > fit_x_max + 1e-9:
            x_eval = np.linspace(orig_x_min, orig_x_max, n_output)
            y_eval = np.polyval(coeffs, x_eval)

            # Clamp extrapolated regions using linear extension from
            # the polynomial's edge value + edge slope, preventing
            # the curve from diverging wildly beyond the data range.
            poly_deriv = np.polyder(coeffs)
            # Left extrapolation
            left_mask = x_eval < fit_x_min
            if left_mask.any():
                y_at_left = float(np.polyval(coeffs, fit_x_min))
                slope_left = float(np.polyval(poly_deriv, fit_x_min))
                y_eval[left_mask] = y_at_left + slope_left * (x_eval[left_mask] - fit_x_min)
            # Right extrapolation
            right_mask = x_eval > fit_x_max
            if right_mask.any():
                y_at_right = float(np.polyval(coeffs, fit_x_max))
                slope_right = float(np.polyval(poly_deriv, fit_x_max))
                y_eval[right_mask] = y_at_right + slope_right * (x_eval[right_mask] - fit_x_max)

            fit["fitted_points"] = [
                {"x": float(xv), "y": float(yv)}
                for xv, yv in zip(x_eval, y_eval)
            ]

    # Add BW-specific metadata
    fit["num_bins_used"] = num_bins
    fit["original_point_count"] = len(axis_coords)

    cleaned_out = list(zip(x_clean.tolist(), y_smooth.tolist()))

    # 6. Peak / best-value point (maximum y in data coordinates)
    fps = fit.get("fitted_points", [])
    if fps:
        peak = max(fps, key=lambda p: p["y"])
        fit["peak_point"] = {"x": round(peak["x"], 4), "y": round(peak["y"], 4)}
    else:
        fit["peak_point"] = None

    logger.info(
        "fit_bw_curve: method=%s  r2=%.4f  rmse=%.4f  bins=%d  "
        "raw=%d  cleaned=%d  peak=%s",
        fit.get("fit_method", "?"),
        fit.get("r_squared", 0),
        fit.get("fit_rmse_on_centerline", 0),
        num_bins,
        len(axis_coords),
        len(cleaned_out),
        fit.get("peak_point"),
    )
    return fit, cleaned_out


# ====================================================================
# 7. Reprojection comparison — raw polyline vs fitted curve
# ====================================================================

def compute_reprojection_rmse(
    raw_pixel_points: List[Tuple[int, int]],
    fitted_data_points: List[Dict[str, float]],
    pixel_to_data_fn,
    data_to_pixel_fn,
) -> Tuple[float, float]:
    """Compute reprojection RMSE for fitted vs raw points.

    raw_pixel_points: original traced pixels [(px, py), ...]
    fitted_data_points: [{"x": dx, "y": dy}, ...]
    pixel_to_data_fn: callable(List[(px,py)]) -> List[(dx,dy)]
    data_to_pixel_fn: callable(List[(dx,dy)]) -> List[(px,py)]

    Returns (raw_reproj_rmse, fitted_reproj_rmse).
    """
    # Raw polyline round-trip error
    raw_data = pixel_to_data_fn(raw_pixel_points)
    raw_back = data_to_pixel_fn(raw_data)
    raw_errors = []
    for (px1, py1), (px2, py2) in zip(raw_pixel_points, raw_back):
        raw_errors.append(math.sqrt((px1 - px2) ** 2 + (py1 - py2) ** 2))
    raw_rmse = math.sqrt(sum(e * e for e in raw_errors) / max(len(raw_errors), 1))

    # Fitted curve: convert data points back to pixel, then measure
    # distance from each reprojected fitted point to nearest raw pixel
    fitted_coords = [(p["x"], p["y"]) for p in fitted_data_points]
    fitted_px = data_to_pixel_fn(fitted_coords)
    if not raw_pixel_points or not fitted_px:
        return (raw_rmse, float("inf"))

    import numpy as np
    raw_arr = np.array(raw_pixel_points, dtype=np.float64)
    fitted_errors = []
    for fpx, fpy in fitted_px:
        dists = np.sqrt((raw_arr[:, 0] - fpx) ** 2 + (raw_arr[:, 1] - fpy) ** 2)
        fitted_errors.append(float(np.min(dists)))
    fitted_rmse = math.sqrt(sum(e * e for e in fitted_errors) / max(len(fitted_errors), 1))

    return (raw_rmse, fitted_rmse)


def build_raw_polyline_fit(
    raw_pixel_points: List[Tuple[int, int]],
    pixel_to_data_fn,
    n_output: int = 300,
) -> Dict[str, Any]:
    """Build a fit_result dict from raw traced polyline (no smoothing).

    This serves as a fallback when the polynomial fit has worse
    reprojection error than the raw trace.
    """
    if len(raw_pixel_points) < 2:
        return {}

    data_pts = pixel_to_data_fn(raw_pixel_points)
    # Sort by x and deduplicate
    sorted_pts = sorted(data_pts, key=lambda p: p[0])

    # Subsample to n_output points
    if len(sorted_pts) > n_output:
        step = max(1, len(sorted_pts) // n_output)
        sorted_pts = sorted_pts[::step]

    xs = [p[0] for p in sorted_pts]
    ys = [p[1] for p in sorted_pts]

    if not xs:
        return {}

    # Compute R² against linear fit to indicate linearity
    import numpy as np
    x_arr = np.array(xs)
    y_arr = np.array(ys)
    if len(x_arr) >= 2:
        coeffs = np.polyfit(x_arr, y_arr, 1)
        y_pred = np.polyval(coeffs, x_arr)
        ss_res = float(np.sum((y_arr - y_pred) ** 2))
        ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    else:
        r2 = 1.0

    return {
        "degree": "raw_polyline",
        "coefficients": None,
        "r_squared": float(r2),
        "fitted_points": [
            {"x": float(x), "y": float(y)} for x, y in zip(xs, ys)
        ],
        "original_point_count": len(raw_pixel_points),
        "equation": "raw traced polyline (no fit)",
        "fit_method": "raw_polyline",
        "fit_rmse_on_centerline": 0.0,
    }
