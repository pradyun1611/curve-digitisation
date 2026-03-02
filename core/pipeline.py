"""
Unified pipeline for curve digitization with metrics and reconstruction.

Combines the extraction, metrics computation, and artifact generation into a
single ``run_pipeline`` call so both CLI and web entry-points share one flow.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.image_processor import CurveDigitizer
from core.metrics import (
    compute_ground_truth_metrics,
    compute_self_consistency_metrics,
    compute_series_regression,
    parse_ground_truth_csv,
    parse_ground_truth_json,
)
from core.reconstruction import (
    build_masks,
    render_overlay_comparison,
    render_reconstructed_plot,
)
from core.scale import compute_affine_mapping, roundtrip_error
from core.calibration import (
    calibrate_from_axis_info,
    build_mapping_from_calibration,
    pixel_to_data as calib_pixel_to_data,
    validate_calibration,
)
from core.types import (
    AxisInfo,
    CurveResult,
    ExtractionResult,
    MappingResult,
    MetricsResult,
)

logger = logging.getLogger(__name__)


def run_pipeline(
    image_path: str,
    axis_info_dict: Dict[str, Any],
    features_dict: Dict[str, Any],
    output_dir: str,
    *,
    job_id: str = "",
    ground_truth_text: Optional[str] = None,
    ground_truth_format: str = "csv",
) -> ExtractionResult:
    """Execute the full digitization -> metrics -> reconstruction pipeline.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    axis_info_dict : dict
        Axis boundaries / units as returned by ``OpenAIClient.extract_axis_info``.
    features_dict : dict
        Curve features as returned by ``OpenAIClient.extract_curve_features``.
    output_dir : str
        Folder where JSON + image artifacts are written.
    job_id : str
        Unique job identifier (auto-generated if blank).
    ground_truth_text : str | None
        Raw text of an optional ground-truth CSV or JSON file.
    ground_truth_format : str
        ``"csv"`` or ``"json"``.

    Returns
    -------
    ExtractionResult
    """
    if not job_id:
        job_id = uuid.uuid4().hex[:12]

    logger.info("[%s] pipeline: START  image=%s", job_id, image_path)
    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Digitize curves (existing logic)
    # ------------------------------------------------------------------
    logger.info("[%s] pipeline: digitizing curves", job_id)
    digitizer = CurveDigitizer(axis_info_dict)
    raw_results: Dict[str, Any] = digitizer.process_curve_image(str(image_path), features_dict)

    # Enrich curve data with pixel coords for metrics
    _enrich_pixel_coords(raw_results, digitizer, image_path, features_dict)

    dims = raw_results.get("image_dimensions", {})
    img_w = dims.get("width", 640)
    img_h = dims.get("height", 480)
    logger.info("[%s] pipeline: image dimensions %dx%d", job_id, img_w, img_h)

    # ------------------------------------------------------------------
    # 2. Compute affine mapping  (FIX: use plot-area, not full image)
    # ------------------------------------------------------------------
    axis_info = AxisInfo.from_dict(axis_info_dict)
    has_mapping = axis_info.has_mapping
    mapping_status = "mapped" if has_mapping else "pixel_only"

    # Extract plot area from digitizer results (critical for correct mapping)
    pa = raw_results.get("plot_area", {})
    pa_left = pa.get("left", 0)
    pa_top = pa.get("top", 0)
    pa_right = pa.get("right", img_w)
    pa_bottom = pa.get("bottom", img_h)
    plot_area_tuple = (pa_left, pa_top, pa_right, pa_bottom)
    plot_w = pa_right - pa_left
    plot_h = pa_bottom - pa_top

    mapping: Optional[MappingResult] = None
    debug_info: Dict[str, Any] = {
        "image_width": img_w,
        "image_height": img_h,
        "plot_area": {"left": pa_left, "top": pa_top,
                      "right": pa_right, "bottom": pa_bottom},
        "plot_area_width": plot_w,
        "plot_area_height": plot_h,
        "has_mapping": has_mapping,
    }

    if has_mapping:
        # Use calibration system for correct plot-area-aware mapping
        calibration = calibrate_from_axis_info(
            axis_info, plot_area_tuple, method="simple",
        )
        mapping = build_mapping_from_calibration(calibration)

        debug_info["pixel_to_data_matrix"] = mapping.pixel_to_data_matrix
        debug_info["data_to_pixel_matrix"] = mapping.data_to_pixel_matrix
        debug_info["x_direction"] = mapping.x_direction
        debug_info["y_direction"] = mapping.y_direction
        debug_info["mapping_frame"] = mapping.frame
        debug_info["calibration_method"] = calibration.method

        logger.info(
            "[%s] pipeline: mapping using plot_area (%d,%d,%d,%d) → "
            "plot_dims=%dx%d  p2d=%s  d2p=%s",
            job_id, pa_left, pa_top, pa_right, pa_bottom,
            plot_w, plot_h,
            mapping.pixel_to_data_matrix, mapping.data_to_pixel_matrix,
        )

        # Round-trip consistency check on all pixel coords
        all_px: List[List[float]] = []
        for cname, cdata in raw_results.get("curves", {}).items():
            if isinstance(cdata, dict):
                pts = (cdata.get("pixel_coords") or [])[:100]
                for p in pts:
                    all_px.append([float(p[0]), float(p[1])])
                logger.debug("[%s] pipeline: curve '%s' contributed %d pts for round-trip check",
                             job_id, cname, len(pts))
        if all_px:
            # Validate using calibration round-trip
            calib_pts = [(p[0], p[1]) for p in all_px]
            rt_mean, rt_p95 = validate_calibration(calib_pts, calibration)
            mapping.mapping_roundtrip_error_mean_px = rt_mean
            mapping.mapping_roundtrip_error_p95_px = rt_p95
            debug_info["mapping_roundtrip_error_mean_px"] = rt_mean
            debug_info["mapping_roundtrip_error_p95_px"] = rt_p95

            logger.info(
                "[%s] pipeline: round-trip error  mean=%.4f px  p95=%.4f px",
                job_id, rt_mean, rt_p95,
            )

            if rt_mean > 2.0:
                mapping_status = "low_confidence"
                logger.warning(
                    "[%s] pipeline: HIGH round-trip error (%.2f px) – "
                    "mapping may be inaccurate", job_id, rt_mean,
                )

    # ------------------------------------------------------------------
    # 3. Build ExtractionResult
    # ------------------------------------------------------------------
    curves: Dict[str, CurveResult] = {}
    for k, v in raw_results.get("curves", {}).items():
        curves[k] = CurveResult.from_dict(v) if isinstance(v, dict) else CurveResult(color=k)

    result = ExtractionResult(
        job_id=job_id,
        image_path=str(image_path),
        image_dimensions=dims,
        axis_info=axis_info,
        curves=curves,
        mapping=mapping,
        debug=debug_info,
        timestamp=datetime.now().isoformat(),
    )

    # ------------------------------------------------------------------
    # 4. Write output dir & save images
    # ------------------------------------------------------------------
    out = Path(output_dir) / job_id
    out.mkdir(parents=True, exist_ok=True)

    # Save original input copy
    from shutil import copy2
    input_img_path = Path(image_path)
    if input_img_path.exists():
        dest_img = out / f"input_{input_img_path.name}"
        copy2(str(input_img_path), str(dest_img))

    # ------------------------------------------------------------------
    # 5. Reconstruction artifacts (clean – no grid)
    # ------------------------------------------------------------------
    curves_dict = raw_results.get("curves", {})

    recon_path = out / "reconstructed_plot.png"
    logger.info("[%s] pipeline: generating reconstructed_plot.png", job_id)
    render_reconstructed_plot(
        curves_dict, axis_info_dict, recon_path,
        has_mapping=has_mapping,
        image_width=img_w,
        image_height=img_h,
        job_id=job_id,
    )
    result.artifacts.append("reconstructed_plot.png")

    overlay_path = out / "overlay_comparison.png"
    logger.info("[%s] pipeline: generating overlay_comparison.png", job_id)
    render_overlay_comparison(
        input_img_path, curves_dict, axis_info_dict, overlay_path,
        has_mapping=has_mapping,
        job_id=job_id,
    )
    result.artifacts.append("overlay_comparison.png")

    # ------------------------------------------------------------------
    # 6. Self-consistency metrics
    # ------------------------------------------------------------------
    logger.info("[%s] pipeline: computing self-consistency metrics", job_id)
    original_mask, recon_mask = build_masks(
        curves_dict, axis_info_dict, input_img_path, has_mapping=has_mapping,
    )

    # Save debug mask images for inspection
    from PIL import Image as _PILImage
    try:
        orig_mask_img = _PILImage.fromarray((original_mask.astype(np.uint8) * 255))
        orig_mask_img.save(str(out / "original_series_mask.png"))
        result.artifacts.append("original_series_mask.png")

        recon_mask_img = _PILImage.fromarray((recon_mask.astype(np.uint8) * 255))
        recon_mask_img.save(str(out / "reconstructed_mask.png"))
        result.artifacts.append("reconstructed_mask.png")

        # mask_diff.png – XOR of both masks (shows mismatch areas)
        diff_mask = np.logical_xor(original_mask, recon_mask)
        diff_img = _PILImage.fromarray((diff_mask.astype(np.uint8) * 255))
        diff_img.save(str(out / "mask_diff.png"))
        result.artifacts.append("mask_diff.png")

        logger.info("[%s] pipeline: saved debug mask images  "
                    "orig_nonzero=%d  recon_nonzero=%d  diff_nonzero=%d",
                    job_id, int(original_mask.sum()), int(recon_mask.sum()),
                    int(diff_mask.sum()))
    except Exception as exc:
        logger.warning("[%s] pipeline: failed to save mask images: %s", job_id, exc)

    metrics = compute_self_consistency_metrics(
        original_mask, recon_mask,
        plot_width=img_w,
        plot_height=img_h,
        mapping_status=mapping_status,
        job_id=job_id,
    )

    # ------------------------------------------------------------------
    # 6b. Per-series regression quality (R² and Pearson R)
    # ------------------------------------------------------------------
    series_regression: Dict[str, Any] = {}
    for cname, cdata in curves_dict.items():
        if not isinstance(cdata, dict) or cdata.get("error"):
            continue
        # Prefer RANSAC-cleaned axis_coords from fit_result > axis_coords > pixel_coords
        fit = cdata.get("fit_result") or {}
        fitted_pts = fit.get("fitted_points", []) if isinstance(fit, dict) else []
        if fitted_pts:
            pts = [(p["x"], p["y"]) for p in fitted_pts]
        else:
            ac = cdata.get("axis_coords") or []
            pts = [(p[0], p[1]) for p in ac] if ac else []
        if pts:
            reg = compute_series_regression(pts, series_name=cname)
            series_regression[cname] = reg
            logger.info("[%s] pipeline: regression '%s'  r2=%.6f  pearson_r=%.6f  n=%d",
                        job_id, cname, reg["r2_score"], reg["pearson_r"],
                        int(reg["n_points"]))
    debug_info["series_regression"] = series_regression

    # ------------------------------------------------------------------
    # 7. Optional ground-truth evaluation
    # ------------------------------------------------------------------
    if ground_truth_text:
        logger.info("[%s] pipeline: computing ground-truth metrics", job_id)
        if ground_truth_format == "json":
            gt_series = parse_ground_truth_json(ground_truth_text)
        else:
            gt_series = parse_ground_truth_csv(ground_truth_text)

        ext_series: Dict[str, list] = {}
        for k, cr in curves.items():
            if cr.axis_coords:
                ext_series[k] = [(p[0], p[1]) for p in cr.axis_coords]
            elif cr.fit_result and cr.fit_result.fitted_points:
                ext_series[k] = [(p["x"], p["y"]) for p in cr.fit_result.fitted_points]

        metrics = compute_ground_truth_metrics(
            ext_series, gt_series, pixel_metrics=metrics, job_id=job_id,
        )

    result.metrics = metrics

    # ------------------------------------------------------------------
    # 8. Persist JSON artifacts
    # ------------------------------------------------------------------
    metrics_json_path = out / "metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    result.artifacts.append("metrics.json")

    report_path = out / "report.json"
    with open(report_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    result.artifacts.append("report.json")

    debug_path = out / "debug.json"
    with open(debug_path, "w") as f:
        json.dump(debug_info, f, indent=2, default=str)
    result.artifacts.append("debug.json")

    elapsed = time.perf_counter() - t_start
    logger.info(
        "[%s] pipeline: DONE in %.2fs  delta_value=%.3f  mapping_status=%s  artifacts=%s",
        job_id, elapsed, metrics.delta_value, mapping_status, result.artifacts,
    )

    return result


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def _enrich_pixel_coords(
    raw_results: Dict[str, Any],
    digitizer: CurveDigitizer,
    image_path: str,
    features_dict: Dict[str, Any],
) -> None:
    """Add ``pixel_coords``, ``raw_pixel_points``, and ``axis_coords`` to raw curve dicts.

    FIX: now passes plot_area to normalize_to_axis so pixel→data
    mapping uses the correct coordinate frame.
    """
    from PIL import Image as PILImage

    try:
        img = PILImage.open(str(image_path)).convert("RGB")
    except Exception:
        return

    width, height = img.size

    # Get plot_area from results (set by process_curve_image)
    pa = raw_results.get("plot_area", {})
    if isinstance(pa, dict) and pa:
        plot_area = (pa.get("left", 0), pa.get("top", 0),
                     pa.get("right", width), pa.get("bottom", height))
    else:
        plot_area = None

    for color_key, cdata in raw_results.get("curves", {}).items():
        if not isinstance(cdata, dict):
            continue

        # --- raw_pixel_points: full resolution, unsorted ---
        if "raw_pixel_points" not in cdata:
            pixels = digitizer.extract_color_pixels(img, cdata.get("color", color_key))
            # Keep all raw points (capped at 2000 for JSON size)
            if len(pixels) > 2000:
                step = max(1, len(pixels) // 2000)
                sampled = pixels[::step]
            else:
                sampled = pixels
            cdata["raw_pixel_points"] = [[int(p[0]), int(p[1])] for p in sampled]

        # --- pixel_coords: sorted by x, down-sampled to 500 ---
        if "pixel_coords" not in cdata:
            raw = cdata.get("raw_pixel_points", [])
            sorted_pts = sorted(raw, key=lambda p: p[0])
            if len(sorted_pts) > 500:
                step = max(1, len(sorted_pts) // 500)
                sorted_pts = sorted_pts[::step]
            cdata["pixel_coords"] = sorted_pts

        # --- axis_coords  (FIX: pass plot_area for correct mapping) ---
        if "axis_coords" not in cdata and cdata.get("pixel_coords"):
            pxs = [(p[0], p[1]) for p in cdata["pixel_coords"]]
            # Use per-curve plot_area if stored, else global
            curve_pa = cdata.get("plot_area")
            if curve_pa and isinstance(curve_pa, (list, tuple)) and len(curve_pa) == 4:
                pa_tuple = tuple(curve_pa)
            else:
                pa_tuple = plot_area
            axis = digitizer.normalize_to_axis(pxs, width, height, pa_tuple)
            if len(axis) > 500:
                step = max(1, len(axis) // 500)
                axis = axis[::step]
            cdata["axis_coords"] = [[round(a[0], 4), round(a[1], 4)] for a in axis]
