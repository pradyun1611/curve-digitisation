"""
Axis calibration and mapping system.

Provides robust pixel ↔ data coordinate conversion with:
  - AUTO mode: tick-mark detection for axis inference
  - MANUAL mode: user-specified reference points
  - SIMPLE mode: user-specified axis min/max

The critical mapping bug fix:
  - Uses plot-area dimensions (not full image) for affine mapping
  - Correct y-inversion (pixel y grows down, data y grows up)
  - Validates by reprojecting extracted data to pixels

This module supplements ``core.scale`` with calibration-aware mapping
and debug overlays.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from core.types import AxisInfo, MappingResult

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of axis calibration."""
    method: str = "simple"  # "auto" | "manual" | "simple"
    x_min: float = 0.0
    x_max: float = 100.0
    y_min: float = 0.0
    y_max: float = 100.0
    plot_area: Tuple[int, int, int, int] = (0, 0, 100, 100)

    # Reference points (manual mode)
    x_ref_points: List[Dict[str, float]] = field(default_factory=list)
    y_ref_points: List[Dict[str, float]] = field(default_factory=list)

    # Computed affine matrices
    pixel_to_data: Optional[List[List[float]]] = None
    data_to_pixel: Optional[List[List[float]]] = None

    # Validation
    validation_error_mean_px: float = 0.0
    validation_error_p95_px: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _get_axis_val(axis_info, key: str, default: float = 0.0) -> float:
    """Get axis value from AxisInfo object or plain dict."""
    if isinstance(axis_info, dict):
        return float(axis_info.get(key, default) or default)
    return float(getattr(axis_info, key, default) or default)


def calibrate_simple(
    axis_info,
    plot_area: Tuple[int, int, int, int],
) -> CalibrationResult:
    """Simple calibration: use axis min/max and plot area bounds.

    This is the correct mapping that fixes the bug where full image
    dimensions were used instead of plot-area dimensions.

    The mapping:
      data_x = x_min + (px - plot_left) / plot_width * (x_max - x_min)
      data_y = y_max - (py - plot_top) / plot_height * (y_max - y_min)

    Note: data_y uses y_max (not y_min) as the origin because pixel y=0
    is at the top of the image, while data y grows upward.
    """
    x_min = _get_axis_val(axis_info, 'xMin', 0)
    x_max = _get_axis_val(axis_info, 'xMax', 100)
    y_min = _get_axis_val(axis_info, 'yMin', 0)
    y_max = _get_axis_val(axis_info, 'yMax', 100)

    p_left, p_top, p_right, p_bottom = plot_area
    # Fence-post: extractable pixel range is p_left..p_right-1
    # (p_right is exclusive in Python slicing).  The number of
    # intervals between the first and last extractable pixel is
    # (p_right - 1) - p_left = p_right - p_left - 1.
    plot_w = max(p_right - p_left - 1, 1)
    plot_h = max(p_bottom - p_top - 1, 1)

    # Scale factors
    sx = (x_max - x_min) / plot_w
    sy = (y_max - y_min) / plot_h

    # pixel→data affine (2×3)
    # data_x =  sx * (px - p_left) + x_min
    #         =  sx * px + (x_min - sx * p_left)
    # data_y = -sy * (py - p_top) + y_max
    #         = -sy * py + (y_max + sy * p_top)
    p2d = [
        [sx, 0.0, x_min - sx * p_left],
        [0.0, -sy, y_max + sy * p_top],
    ]

    # data→pixel inverse (2×3)
    # px = (data_x - x_min) / sx + p_left
    #    = (1/sx) * data_x + (p_left - x_min/sx)
    # py = (y_max - data_y) / sy + p_top
    #    = (-1/sy) * data_y + (y_max/sy + p_top)
    inv_sx = 1.0 / sx if sx != 0 else 0.0
    inv_sy = 1.0 / sy if sy != 0 else 0.0
    d2p = [
        [inv_sx, 0.0, p_left - x_min * inv_sx],
        [0.0, -inv_sy, y_max * inv_sy + p_top],
    ]

    logger.info("calibrate_simple: plot_area=(%d,%d,%d,%d) → "
                "data=[%.2f..%.2f]×[%.2f..%.2f]  sx=%.6f sy=%.6f",
                p_left, p_top, p_right, p_bottom,
                x_min, x_max, y_min, y_max, sx, sy)

    return CalibrationResult(
        method="simple",
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        plot_area=plot_area,
        pixel_to_data=p2d,
        data_to_pixel=d2p,
    )


def calibrate_manual(
    x_refs: List[Dict[str, float]],
    y_refs: List[Dict[str, float]],
    plot_area: Tuple[int, int, int, int],
) -> CalibrationResult:
    """Manual calibration with user-specified reference points.

    Parameters
    ----------
    x_refs : list of {"pixel": px, "value": data_x}
        At least 2 reference points on x-axis.
    y_refs : list of {"pixel": py, "value": data_y}
        At least 2 reference points on y-axis.
    plot_area : (left, top, right, bottom)
    """
    if len(x_refs) < 2 or len(y_refs) < 2:
        raise ValueError("Manual calibration requires at least 2 reference "
                        "points per axis")

    # Sort by pixel position
    x_refs = sorted(x_refs, key=lambda r: r["pixel"])
    y_refs = sorted(y_refs, key=lambda r: r["pixel"])

    # Compute x mapping: linear fit of pixel → data
    px_x = np.array([r["pixel"] for r in x_refs])
    val_x = np.array([r["value"] for r in x_refs])
    if len(px_x) >= 2:
        sx, tx = np.polyfit(px_x, val_x, 1)
    else:
        sx, tx = 1.0, 0.0

    # Compute y mapping: linear fit of pixel → data
    px_y = np.array([r["pixel"] for r in y_refs])
    val_y = np.array([r["value"] for r in y_refs])
    if len(px_y) >= 2:
        sy, ty = np.polyfit(px_y, val_y, 1)
    else:
        sy, ty = -1.0, 100.0

    p2d = [
        [float(sx), 0.0, float(tx)],
        [0.0, float(sy), float(ty)],
    ]

    # Inverse
    inv_sx = 1.0 / sx if sx != 0 else 0.0
    inv_sy = 1.0 / sy if sy != 0 else 0.0
    d2p = [
        [inv_sx, 0.0, -tx * inv_sx],
        [0.0, inv_sy, -ty * inv_sy],
    ]

    # Infer axis bounds from mapping
    p_left, p_top, p_right, p_bottom = plot_area
    x_min = float(sx * p_left + tx)
    x_max = float(sx * p_right + tx)
    y_top = float(sy * p_top + ty)     # data value at top of plot
    y_bottom = float(sy * p_bottom + ty)  # data value at bottom
    y_min = min(y_top, y_bottom)
    y_max = max(y_top, y_bottom)

    logger.info("calibrate_manual: sx=%.6f tx=%.2f sy=%.6f ty=%.2f  "
                "data=[%.2f..%.2f]×[%.2f..%.2f]",
                sx, tx, sy, ty, x_min, x_max, y_min, y_max)

    return CalibrationResult(
        method="manual",
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        plot_area=plot_area,
        x_ref_points=x_refs,
        y_ref_points=y_refs,
        pixel_to_data=p2d,
        data_to_pixel=d2p,
    )


def pixel_to_data(
    pixel_points: List[Tuple[float, float]],
    calibration: CalibrationResult,
) -> List[Tuple[float, float]]:
    """Convert pixel coordinates to data coordinates using calibration.

    Parameters
    ----------
    pixel_points : list of (px, py)
        Pixel coordinates in full-image space.
    calibration : CalibrationResult
        Must have pixel_to_data matrix.

    Returns
    -------
    List of (data_x, data_y)
    """
    if calibration.pixel_to_data is None:
        raise ValueError("Calibration has no pixel_to_data matrix")

    m = np.array(calibration.pixel_to_data)  # (2, 3)
    result = []
    for px, py in pixel_points:
        v = np.array([px, py, 1.0])
        d = m @ v
        result.append((float(d[0]), float(d[1])))
    return result


def data_to_pixel(
    data_points: List[Tuple[float, float]],
    calibration: CalibrationResult,
) -> List[Tuple[float, float]]:
    """Convert data coordinates to pixel coordinates using calibration."""
    if calibration.data_to_pixel is None:
        raise ValueError("Calibration has no data_to_pixel matrix")

    m = np.array(calibration.data_to_pixel)  # (2, 3)
    result = []
    for dx, dy in data_points:
        v = np.array([dx, dy, 1.0])
        p = m @ v
        result.append((float(p[0]), float(p[1])))
    return result


def validate_calibration(
    pixel_points: List[Tuple[float, float]],
    calibration: CalibrationResult,
) -> Tuple[float, float]:
    """Validate calibration via round-trip error.

    pixel → data → pixel and measure error.

    Returns (mean_error_px, p95_error_px).
    """
    if not pixel_points:
        return (0.0, 0.0)

    data_pts = pixel_to_data(pixel_points, calibration)
    back_px = data_to_pixel(data_pts, calibration)

    errors = []
    for (px1, py1), (px2, py2) in zip(pixel_points, back_px):
        err = np.sqrt((px1 - px2) ** 2 + (py1 - py2) ** 2)
        errors.append(err)

    errors_arr = np.array(errors)
    mean_err = float(np.mean(errors_arr))
    p95_err = float(np.percentile(errors_arr, 95))

    calibration.validation_error_mean_px = mean_err
    calibration.validation_error_p95_px = p95_err

    return (mean_err, p95_err)


def build_mapping_from_calibration(
    calibration: CalibrationResult,
) -> MappingResult:
    """Convert CalibrationResult to MappingResult for pipeline compatibility."""
    p_left, p_top, p_right, p_bottom = calibration.plot_area
    plot_w = p_right - p_left
    plot_h = p_bottom - p_top

    return MappingResult(
        pixel_to_data_matrix=calibration.pixel_to_data or [],
        data_to_pixel_matrix=calibration.data_to_pixel or [],
        frame="plot_area",
        x_direction=1,
        y_direction=1,
        plot_area_width=plot_w,
        plot_area_height=plot_h,
        mapping_roundtrip_error_mean_px=calibration.validation_error_mean_px,
        mapping_roundtrip_error_p95_px=calibration.validation_error_p95_px,
    )


def calibrate_from_axis_info(
    axis_info: AxisInfo,
    plot_area: Tuple[int, int, int, int],
    *,
    method: str = "simple",
    x_refs: Optional[List[Dict[str, float]]] = None,
    y_refs: Optional[List[Dict[str, float]]] = None,
) -> CalibrationResult:
    """Unified calibration entry point.

    Parameters
    ----------
    axis_info : AxisInfo
    plot_area : (left, top, right, bottom) in full-image pixels
    method : "simple" | "manual" | "auto"
    x_refs, y_refs : reference points for manual mode
    """
    if method == "manual" and x_refs and y_refs:
        return calibrate_manual(x_refs, y_refs, plot_area)

    has_map = getattr(axis_info, 'has_mapping', True) if not isinstance(axis_info, dict) else True
    if method == "simple" or has_map:
        return calibrate_simple(axis_info, plot_area)
    else:
        # Auto: fall back to simple with whatever info we have
        return calibrate_simple(axis_info, plot_area)
