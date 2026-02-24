"""
Quality metrics module for curve digitization evaluation.

Provides two evaluation modes:
  A) Self-consistency  – compare reconstructed curve (pixel space) vs original
     extracted stroke mask.  No ground truth needed.
  B) Ground-truth      – compare extracted data-space coordinates against a
     reference CSV / JSON.

Primary output is ``MetricsResult`` (see core.types).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

try:
    from skimage.metrics import structural_similarity as _ssim_fn
except ImportError:  # pragma: no cover
    _ssim_fn = None  # type: ignore[assignment]

from core.types import MetricsResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# public API
# ------------------------------------------------------------------

def compute_self_consistency_metrics(
    original_mask: np.ndarray,
    reconstructed_mask: np.ndarray,
    plot_width: int,
    plot_height: int,
    *,
    mapping_status: str = "pixel_only",
    job_id: str = "",
) -> MetricsResult:
    """Compute self-consistency metrics between two binary masks.

    Parameters
    ----------
    original_mask : np.ndarray (H, W) bool / uint8
        Binary mask of the *original* extracted strokes.
    reconstructed_mask : np.ndarray (H, W) bool / uint8
        Binary mask rendered from the extracted coordinates.
    plot_width, plot_height : int
        Dimensions of the plot-area image (used for normalisation).
    mapping_status : str
        ``"mapped"`` or ``"pixel_only"``.
    job_id : str
        For log correlation.

    Returns
    -------
    MetricsResult
    """
    logger.info("[%s] metrics: self-consistency computation start", job_id)
    t0 = time.perf_counter()

    orig = _to_bool(original_mask)
    recon = _to_bool(reconstructed_mask)

    # Sanity: log mask non-zero counts early for debugging
    orig_nz = int(orig.sum())
    recon_nz = int(recon.sum())
    logger.info("[%s] metrics: mask shapes orig=%s recon=%s  "
                "orig_nonzero=%d  recon_nonzero=%d",
                job_id, orig.shape, recon.shape, orig_nz, recon_nz)
    if orig_nz == 0:
        logger.warning("[%s] metrics: original mask is EMPTY – all metrics will be trivial", job_id)
    if recon_nz == 0:
        logger.warning("[%s] metrics: reconstructed mask is EMPTY – delta will be worst-case", job_id)
    if orig_nz > 0 and recon_nz > 0:
        overlap = int(np.logical_and(orig, recon).sum())
        overlap_pct = 100.0 * overlap / min(orig_nz, recon_nz)
        logger.info("[%s] metrics: mask overlap=%d (%.1f%% of smaller mask)",
                    job_id, overlap, overlap_pct)
        if overlap_pct > 95.0:
            logger.warning("[%s] metrics: masks overlap >95%% – reconstructed mask may be "
                           "derived from the same source as original", job_id)

    # 1. SSIM ----------------------------------------------------------
    ssim_val: Optional[float] = None
    if _ssim_fn is not None:
        try:
            img_a = orig.astype(np.uint8) * 255
            img_b = recon.astype(np.uint8) * 255
            # Ensure same shape
            if img_a.shape == img_b.shape and img_a.size > 0:
                # win_size must be odd and <= smallest image dimension
                min_dim = min(img_a.shape[0], img_a.shape[1])
                win_size = min(7, min_dim)
                if win_size % 2 == 0:
                    win_size = max(win_size - 1, 3)
                if win_size >= 3 and min_dim >= win_size:
                    ssim_val = float(
                        _ssim_fn(
                            img_a, img_b,
                            data_range=255,
                            win_size=win_size,
                        )
                    )
                    logger.info("[%s] metrics: SSIM computed = %.4f  win_size=%d",
                                job_id, ssim_val, win_size)
                else:
                    logger.warning("[%s] metrics: image too small for SSIM (min_dim=%d)", job_id, min_dim)
            else:
                logger.warning("[%s] metrics: shape mismatch for SSIM: %s vs %s",
                               job_id, img_a.shape, img_b.shape)
        except Exception as exc:
            logger.warning("[%s] SSIM computation failed: %s", job_id, exc)

    # 2. IoU (Jaccard) -------------------------------------------------
    intersection = np.logical_and(orig, recon).sum()
    union = np.logical_or(orig, recon).sum()
    iou = float(intersection / union) if union > 0 else 0.0

    # 3. Precision / Recall -------------------------------------------
    tp = intersection
    fp = np.logical_and(recon, ~orig).sum()
    fn = np.logical_and(orig, ~recon).sum()
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # 4. Symmetric chamfer distance -----------------------------------
    delta_mean, delta_p95 = _symmetric_chamfer(orig, recon)

    # 5. Dilated (thickness-tolerant) delta ---------------------------
    delta_dilated_mean = _dilated_delta(orig, recon, radius=2)

    # 6. Normalised delta ---------------------------------------------
    max_dim = max(plot_width, plot_height, 1)
    delta_norm = delta_mean / max_dim

    notes = ""
    if mapping_status == "pixel_only":
        notes = "Warning: mapping_status is pixel_only – metrics are in pixel space only."
    if orig_nz == 0 or recon_nz == 0:
        delta_mean = float('nan')
        delta_p95 = float('nan')
        delta_norm = float('nan')
        notes += " Warning: one or both masks empty – delta is NaN."
        notes = notes.strip()

    elapsed = time.perf_counter() - t0
    logger.info(
        "[%s] metrics: done in %.3fs  delta_value=%.3f  delta_dilated=%.3f  "
        "iou=%.4f  ssim=%s  precision=%.4f  recall=%.4f",
        job_id, elapsed, delta_mean, delta_dilated_mean, iou, ssim_val,
        precision, recall,
    )

    return MetricsResult(
        mode="self_consistency",
        delta_value=delta_mean,
        delta_pixels_mean=delta_mean,
        delta_pixels_p95=delta_p95,
        delta_norm=delta_norm,
        iou=iou,
        precision=precision,
        recall=recall,
        ssim=ssim_val,
        mapping_status=mapping_status,
        notes=notes,
    )


def compute_ground_truth_metrics(
    extracted_series: Dict[str, List[Tuple[float, float]]],
    ground_truth: Dict[str, List[Tuple[float, float]]],
    *,
    pixel_metrics: Optional[MetricsResult] = None,
    job_id: str = "",
) -> MetricsResult:
    """Compare extracted data-space coordinates against ground truth.

    Parameters
    ----------
    extracted_series : dict[str, list[(x,y)]]
        Extracted series keyed by name/color.
    ground_truth : dict[str, list[(x,y)]]
        Ground-truth series (same key convention or auto-matched).
    pixel_metrics : MetricsResult | None
        Pre-computed self-consistency result to merge pixel-space fields.
    job_id : str

    Returns
    -------
    MetricsResult
    """
    logger.info("[%s] metrics: ground-truth computation start", job_id)
    t0 = time.perf_counter()

    # Match extracted series to GT series
    matched_pairs = _match_series(extracted_series, ground_truth)

    all_rmse: List[float] = []
    all_mae: List[float] = []
    all_max_abs: List[float] = []

    for ext_pts, gt_pts in matched_pairs:
        rmse, mae, max_abs = _series_error(ext_pts, gt_pts)
        all_rmse.append(rmse)
        all_mae.append(mae)
        all_max_abs.append(max_abs)

    rmse_y = float(np.mean(all_rmse)) if all_rmse else 0.0
    mae_y = float(np.mean(all_mae)) if all_mae else 0.0
    max_abs_y = float(np.max(all_max_abs)) if all_max_abs else 0.0

    base = pixel_metrics or MetricsResult()
    base.mode = "ground_truth"
    base.rmse_y = rmse_y
    base.mae_y = mae_y
    base.max_abs_y = max_abs_y
    base.delta_value_data = rmse_y
    base.mapping_status = "mapped"
    if not base.notes:
        base.notes = ""
    base.notes = (base.notes + " Ground-truth evaluation included.").strip()

    elapsed = time.perf_counter() - t0
    logger.info(
        "[%s] metrics: ground-truth done in %.3fs  rmse_y=%.4f  mae_y=%.4f",
        job_id, elapsed, rmse_y, mae_y,
    )
    return base


def parse_ground_truth_csv(csv_text: str) -> Dict[str, List[Tuple[float, float]]]:
    """Parse a ground-truth CSV into series dict.

    Expected columns: ``series`` (optional), ``x``, ``y``
    """
    import csv
    import io

    reader = csv.DictReader(io.StringIO(csv_text))
    series_map: Dict[str, List[Tuple[float, float]]] = {}
    for row in reader:
        s = row.get("series", "default") or "default"
        try:
            x = float(row["x"])
            y = float(row["y"])
        except (KeyError, ValueError):
            continue
        series_map.setdefault(s, []).append((x, y))
    return series_map


def parse_ground_truth_json(json_text: str) -> Dict[str, List[Tuple[float, float]]]:
    """Parse ground-truth JSON.

    Accepts: ``{"series_name": [[x,y], ...], ...}``
    or       ``[{"series":"name","x":1,"y":2}, ...]``
    """
    import json as _json

    data = _json.loads(json_text)
    result: Dict[str, List[Tuple[float, float]]] = {}
    if isinstance(data, dict):
        for name, points in data.items():
            result[name] = [(float(p[0]), float(p[1])) for p in points]
    elif isinstance(data, list):
        for row in data:
            s = row.get("series", "default")
            result.setdefault(s, []).append((float(row["x"]), float(row["y"])))
    return result


# ------------------------------------------------------------------
# private helpers
# ------------------------------------------------------------------

def _to_bool(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != bool:
        return mask.astype(bool)
    return mask


def _pixel_deviation(orig: np.ndarray, recon: np.ndarray) -> Tuple[float, float]:
    """Mean and P95 distance of original-mask pixels to nearest reconstructed pixel."""
    if not recon.any():
        h, w = orig.shape[:2]
        diag = float(np.sqrt(h ** 2 + w ** 2))
        return (diag, diag) if orig.any() else (0.0, 0.0)

    dt = distance_transform_edt(~recon)
    if not orig.any():
        return (0.0, 0.0)

    distances = dt[orig]
    mean_d = float(np.mean(distances))
    p95_d = float(np.percentile(distances, 95))
    return (mean_d, p95_d)


def _symmetric_chamfer(orig: np.ndarray, recon: np.ndarray) -> Tuple[float, float]:
    """Symmetric (bidirectional) chamfer distance.

    Returns (mean, p95) averaged over both directions:
      d1 = distance from each orig pixel to nearest recon pixel
      d2 = distance from each recon pixel to nearest orig pixel
    """
    if not orig.any() or not recon.any():
        h, w = orig.shape[:2]
        diag = float(np.sqrt(h ** 2 + w ** 2))
        if not orig.any() and not recon.any():
            return (0.0, 0.0)
        return (diag, diag)

    # orig -> nearest recon
    dt_recon = distance_transform_edt(~recon)
    d1 = dt_recon[orig]

    # recon -> nearest orig
    dt_orig = distance_transform_edt(~orig)
    d2 = dt_orig[recon]

    mean_d = (float(np.mean(d1)) + float(np.mean(d2))) / 2.0
    p95_d = (float(np.percentile(d1, 95)) + float(np.percentile(d2, 95))) / 2.0

    logger.debug("symmetric_chamfer: d1_mean=%.3f d2_mean=%.3f  "
                 "d1_p95=%.3f d2_p95=%.3f",
                 float(np.mean(d1)), float(np.mean(d2)),
                 float(np.percentile(d1, 95)), float(np.percentile(d2, 95)))
    return (mean_d, p95_d)


def _dilated_delta(orig: np.ndarray, recon: np.ndarray, radius: int = 2) -> float:
    """Symmetric chamfer on dilated masks – tolerant to stroke thickness."""
    from scipy.ndimage import binary_dilation

    struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    orig_d = binary_dilation(orig, structure=struct)
    recon_d = binary_dilation(recon, structure=struct)

    mean_d, _ = _symmetric_chamfer(orig_d, recon_d)
    return mean_d


def _match_series(
    extracted: Dict[str, List[Tuple[float, float]]],
    ground_truth: Dict[str, List[Tuple[float, float]]],
) -> List[Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]]:
    """Pair extracted series with ground-truth by key or minimal RMSE."""
    pairs: List[Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]] = []
    used_gt_keys: set = set()

    # exact key match
    for ek, ev in extracted.items():
        if ek in ground_truth and ek not in used_gt_keys:
            pairs.append((ev, ground_truth[ek]))
            used_gt_keys.add(ek)

    # remaining: best-RMSE match
    remaining_ext = {k: v for k, v in extracted.items() if k not in {p[0] for p in pairs}}  # type: ignore
    remaining_gt = {k: v for k, v in ground_truth.items() if k not in used_gt_keys}

    # Re-check remaining extraction keys
    remaining_ext = {}
    for k, v in extracted.items():
        already_paired = any(id(v) == id(p[0]) for p in pairs)
        if not already_paired:
            remaining_ext[k] = v

    for ek, ev in remaining_ext.items():
        best_gk = None
        best_rmse = float("inf")
        for gk, gv in remaining_gt.items():
            if gk in used_gt_keys:
                continue
            rmse, _, _ = _series_error(ev, gv)
            if rmse < best_rmse:
                best_rmse = rmse
                best_gk = gk
        if best_gk is not None:
            pairs.append((ev, remaining_gt[best_gk]))
            used_gt_keys.add(best_gk)

    return pairs


def _series_error(
    ext: List[Tuple[float, float]],
    gt: List[Tuple[float, float]],
) -> Tuple[float, float, float]:
    """RMSE, MAE, MaxAbsError on a common x-grid."""
    if not ext or not gt:
        return (0.0, 0.0, 0.0)

    ext_arr = np.array(sorted(ext, key=lambda p: p[0]))
    gt_arr = np.array(sorted(gt, key=lambda p: p[0]))

    # Common x range
    x_lo = max(ext_arr[0, 0], gt_arr[0, 0])
    x_hi = min(ext_arr[-1, 0], gt_arr[-1, 0])
    if x_lo >= x_hi:
        return (0.0, 0.0, 0.0)

    grid = np.linspace(x_lo, x_hi, num=200)
    ext_interp = np.interp(grid, ext_arr[:, 0], ext_arr[:, 1])
    gt_interp = np.interp(grid, gt_arr[:, 0], gt_arr[:, 1])

    diff = ext_interp - gt_interp
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    return (rmse, mae, max_abs)


def compute_series_regression(
    points: List[Tuple[float, float]],
    *,
    series_name: str = "",
) -> Dict[str, float]:
    """Compute linear regression quality metrics for a single series.

    Returns ``{"r2_score": ..., "pearson_r": ..., "n_points": ...}``.

    Uses a **robust** pipeline:
      1. Sort by x, deduplicate, remove NaN/Inf
      2. Compute Pearson R (linear correlation)
      3. Compute R² from sklearn linear regression if available,
         otherwise from np.polyfit degree 1.

    For near-linear series (e.g. a straight blue line), R² should be ~0.99.
    """
    from scipy.stats import pearsonr

    result: Dict[str, float] = {"r2_score": 0.0, "pearson_r": 0.0, "n_points": 0}

    if not points or len(points) < 3:
        logger.debug("compute_series_regression: '%s' has too few points (%d)",
                      series_name, len(points) if points else 0)
        return result

    # 1) Clean: sort, deduplicate, remove NaN/Inf
    arr = np.array(sorted(set(points), key=lambda p: p[0]), dtype=float)
    valid = np.isfinite(arr).all(axis=1)
    arr = arr[valid]
    if len(arr) < 3:
        return result

    xs = arr[:, 0]
    ys = arr[:, 1]
    result["n_points"] = float(len(xs))

    # 2) Pearson R
    try:
        pr, _ = pearsonr(xs, ys)
        result["pearson_r"] = float(pr) if np.isfinite(pr) else 0.0
    except Exception:
        result["pearson_r"] = 0.0

    # 3) Linear R² via np.polyfit degree 1
    try:
        coeffs = np.polyfit(xs, ys, 1)
        y_pred = np.polyval(coeffs, xs)
        ss_res = float(np.sum((ys - y_pred) ** 2))
        ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        result["r2_score"] = float(r2) if np.isfinite(r2) else 0.0
    except Exception:
        result["r2_score"] = 0.0

    logger.info("compute_series_regression: '%s'  n=%d  r2=%.6f  pearson_r=%.6f",
                series_name, int(result["n_points"]),
                result["r2_score"], result["pearson_r"])
    return result
