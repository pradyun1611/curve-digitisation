"""
Affine mapping between pixel coordinates and data coordinates.

All pixel coordinates are in **plot-area-local** space (origin = top-left of
the plot-area crop).  The affine is a 2×3 matrix:

    [data_x]   [a  0  tx] [px]
    [data_y] = [0  d  ty] [py]
                           [ 1]

Because axis-aligned charts have independent x/y scaling the off-diagonal
elements are zero.  ``y_direction`` encodes the pixel-y flip (pixel y grows
downward, data y grows upward → ``d`` is negative by default).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.types import AxisInfo, MappingResult

logger = logging.getLogger(__name__)


def compute_affine_mapping(
    axis_info: AxisInfo,
    plot_width: int,
    plot_height: int,
    *,
    job_id: str = "",
) -> MappingResult:
    """Build forward (pixel→data) and inverse (data→pixel) affine matrices.

    Parameters
    ----------
    axis_info : AxisInfo
        Must have ``has_mapping == True`` (all four bounds known).
    plot_width, plot_height : int
        Dimensions of the plot-area crop in pixels.
    job_id : str
        For log correlation.

    Returns
    -------
    MappingResult
    """
    x_min = float(axis_info.xMin)  # type: ignore[arg-type]
    x_max = float(axis_info.xMax)  # type: ignore[arg-type]
    y_min = float(axis_info.yMin)  # type: ignore[arg-type]
    y_max = float(axis_info.yMax)  # type: ignore[arg-type]

    # Scale factors  ------------------------------------------------
    # pixel x: 0 → plot_width   maps to  data x: x_min → x_max
    sx = (x_max - x_min) / max(plot_width, 1)
    # pixel y: 0 (top) → plot_height (bottom)  maps to  data y: y_max → y_min
    # so  data_y = y_max - (py / plot_height) * (y_max - y_min)
    #            = -sy * py + y_max          where sy = (y_max - y_min)/plot_height
    sy_abs = (y_max - y_min) / max(plot_height, 1)

    # pixel→data  (2×3)
    # data_x =  sx * px + x_min
    # data_y = -sy * py + y_max
    p2d = [
        [sx, 0.0, x_min],
        [0.0, -sy_abs, y_max],
    ]

    # data→pixel  (inverse, 2×3)
    # px = (data_x - x_min) / sx
    # py = (y_max - data_y) / sy_abs
    inv_sx = 1.0 / sx if sx != 0 else 0.0
    inv_sy = 1.0 / sy_abs if sy_abs != 0 else 0.0
    d2p = [
        [inv_sx, 0.0, -x_min * inv_sx],
        [0.0, -inv_sy, y_max * inv_sy],
    ]

    logger.info(
        "[%s] scale: affine computed  plot_area=%dx%d  data=[%.4f..%.4f]x[%.4f..%.4f]  "
        "sx=%.6f  sy=%.6f",
        job_id, plot_width, plot_height, x_min, x_max, y_min, y_max, sx, sy_abs,
    )
    logger.info("[%s] scale: pixel_to_data_matrix = %s", job_id, p2d)
    logger.info("[%s] scale: data_to_pixel_matrix = %s", job_id, d2p)
    logger.info("[%s] scale: x_direction=%d  y_direction=%d  frame=plot_area",
                job_id, 1, 1)

    return MappingResult(
        pixel_to_data_matrix=p2d,
        data_to_pixel_matrix=d2p,
        frame="plot_area",
        x_direction=1,
        y_direction=1,
        plot_area_width=plot_width,
        plot_area_height=plot_height,
    )


# ------------------------------------------------------------------
# Conversion helpers
# ------------------------------------------------------------------

def pixels_to_data(
    pixel_points: List[List[float]],
    mapping: MappingResult,
) -> List[List[float]]:
    """Convert plot-area pixel coords → data coords using the affine."""
    m = np.array(mapping.pixel_to_data_matrix)  # (2,3)
    out: List[List[float]] = []
    for px, py in pixel_points:
        v = np.array([px, py, 1.0])
        d = m @ v
        out.append([round(float(d[0]), 6), round(float(d[1]), 6)])
    return out


def data_to_pixels(
    data_points: List[List[float]],
    mapping: MappingResult,
) -> List[List[float]]:
    """Convert data coords → plot-area pixel coords using the inverse affine."""
    m = np.array(mapping.data_to_pixel_matrix)  # (2,3)
    out: List[List[float]] = []
    for dx, dy in data_points:
        v = np.array([dx, dy, 1.0])
        p = m @ v
        out.append([round(float(p[0]), 4), round(float(p[1]), 4)])
    return out


def roundtrip_error(
    pixel_points: List[List[float]],
    mapping: MappingResult,
) -> Tuple[float, float]:
    """Measure pixel→data→pixel round-trip error.

    Returns
    -------
    (mean_error_px, p95_error_px)
    """
    if not pixel_points:
        return (0.0, 0.0)

    data_pts = pixels_to_data(pixel_points, mapping)
    back_px = data_to_pixels(data_pts, mapping)

    dists = []
    for (ox, oy), (bx, by) in zip(pixel_points, back_px):
        dists.append(((ox - bx) ** 2 + (oy - by) ** 2) ** 0.5)

    arr = np.array(dists)
    return (float(np.mean(arr)), float(np.percentile(arr, 95)))
