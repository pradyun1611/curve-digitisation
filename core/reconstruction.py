"""
Reconstruction module – builds visual artifacts for quality evaluation.

Artifacts produced:
  - ``reconstructed_plot.png``   – clean re-plot from fitted data (no grid)
  - ``overlay_comparison.png``   – original image with RECONSTRUCTED polylines
    (uses fitted/processed points, NOT raw pixels, to avoid double-draw)

The overlay draws the pipeline's *reconstruction* of the curve – i.e. the
polynomial fit converted back to pixel space.  This prevents the visual
"double-draw" artefact that occurs when raw extracted pixels are re-drawn
on top of the same pixels in the original image.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# High-contrast overlay colours – chosen to be clearly distinct from common
# series colours (red, blue, green, black) so the overlay layer is obvious.
# BGR for OpenCV, RGB for PIL.
_OVERLAY_COLORS_BGR = [
    (0, 255, 0),      # lime green
    (255, 255, 0),    # cyan
    (0, 255, 255),    # yellow
    (255, 0, 255),    # magenta
    (0, 165, 255),    # orange
    (255, 200, 0),    # light blue
    (0, 200, 200),    # dark yellow
]

_OVERLAY_COLORS_RGB = [
    (0, 255, 0),      # lime green
    (0, 255, 255),    # cyan
    (255, 255, 0),    # yellow
    (255, 0, 255),    # magenta
    (255, 165, 0),    # orange
    (0, 200, 255),    # light blue
    (200, 200, 0),    # dark yellow
]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def render_reconstructed_plot(
    curves: Dict[str, Any],
    axis_info: Dict[str, Any],
    output_path: Path,
    *,
    has_mapping: bool = True,
    image_width: int = 640,
    image_height: int = 480,
    job_id: str = "",
) -> Path:
    """Render extracted series into a clean matplotlib plot (NO grid lines).

    Uses fitted_points (smooth polynomial) as primary source for the cleanest
    readable curve.  Falls back to axis_coords or pixel_coords.

    Returns the path to the saved image.
    """
    logger.info("[%s] reconstruction: rendering reconstructed_plot start  "
                "has_mapping=%s  canvas=%dx%d", job_id, has_mapping, image_width, image_height)
    t0 = time.perf_counter()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    # Use series colours – avoid black to prevent confusion with grid/axis
    color_cycle = ["red", "blue", "green", "orange", "purple", "brown", "magenta"]
    idx = 0
    plotted_any = False

    for name, cdata in curves.items():
        if not isinstance(cdata, dict) or cdata.get("error"):
            continue

        xs, ys = _get_series_xy(cdata, has_mapping, image_width, image_height)
        if not xs:
            logger.debug("[%s] reconstruction: curve '%s' has no plottable data", job_id, name)
            continue

        label_text = cdata.get("label", name)
        c = color_cycle[idx % len(color_cycle)]
        # Solid line, no dashes, no black
        ax.plot(xs, ys, color=c, linewidth=2, linestyle="-", label=label_text)
        logger.info("[%s] reconstruction: plotted '%s' (%d pts) color=%s  "
                    "x=[%.2f..%.2f] y=[%.2f..%.2f]",
                    job_id, label_text, len(xs), c,
                    min(xs), max(xs), min(ys), max(ys))
        idx += 1
        plotted_any = True

    if has_mapping:
        ax.set_xlabel(axis_info.get("xUnit", "X"))
        ax.set_ylabel(axis_info.get("yUnit", "Y"))
        x_lo, x_hi = axis_info.get("xMin"), axis_info.get("xMax")
        y_lo, y_hi = axis_info.get("yMin"), axis_info.get("yMax")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        logger.info("[%s] reconstruction: axis limits x=[%s..%s] y=[%s..%s]",
                    job_id, x_lo, x_hi, y_lo, y_hi)
    else:
        ax.invert_yaxis()
        ax.set_xlabel("pixel X")
        ax.set_ylabel("pixel Y")

    # ---- CLEAN: absolutely no grid, no black dashed lines ----
    ax.grid(False)
    # Remove minor ticks that could appear as dotted lines
    ax.minorticks_off()

    if plotted_any:
        ax.legend(loc="best", fontsize=8, framealpha=0.7)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)

    elapsed = time.perf_counter() - t0
    logger.info("[%s] reconstruction: reconstructed_plot done in %.3fs -> %s",
                job_id, elapsed, output_path)
    return output_path


def render_overlay_comparison(
    original_image_path: Path,
    curves: Dict[str, Any],
    axis_info: Dict[str, Any],
    output_path: Path,
    *,
    has_mapping: bool = True,
    job_id: str = "",
) -> Path:
    """Overlay RECONSTRUCTED polylines on the original image.

    Uses **reconstructed** points (fitted polynomial → pixel conversion)
    rather than raw extracted pixels.  This avoids the "double-draw" artefact
    where the overlay traces exactly over the original curve.

    Overlay colours are bright lime/cyan/yellow so they stand out clearly
    against the original chart background.

    Returns the path to the saved overlay image.
    """
    logger.info("[%s] reconstruction: rendering overlay_comparison start  image=%s",
                job_id, original_image_path)
    t0 = time.perf_counter()

    img = Image.open(str(original_image_path)).convert("RGB")
    img_array = np.array(img)
    width, height = img.size
    logger.info("[%s] reconstruction: overlay canvas %dx%d", job_id, width, height)

    try:
        import cv2
        use_cv2 = True
    except ImportError:
        use_cv2 = False

    canvas = img_array.copy()
    colour_idx = 0

    for name, cdata in curves.items():
        if not isinstance(cdata, dict) or cdata.get("error"):
            continue

        # Use RECONSTRUCTED points (fitted/processed), NOT raw pixels
        pixel_pts = _get_reconstructed_pixel_points(
            cdata, axis_info, width, height, has_mapping,
        )
        if len(pixel_pts) < 2:
            logger.debug("[%s] reconstruction: overlay curve '%s' skipped (<2 pts)", job_id, name)
            continue
        logger.info("[%s] reconstruction: overlay curve '%s' with %d reconstructed pixel pts",
                    job_id, name, len(pixel_pts))

        if use_cv2:
            import cv2
            pts_int = np.array(pixel_pts, dtype=np.int32).reshape(-1, 1, 2)
            bgr = _OVERLAY_COLORS_BGR[colour_idx % len(_OVERLAY_COLORS_BGR)]
            cv2.polylines(canvas, [pts_int], isClosed=False,
                          color=bgr, thickness=2, lineType=cv2.LINE_AA)
        else:
            from PIL import ImageDraw
            overlay_img = Image.fromarray(canvas)
            draw = ImageDraw.Draw(overlay_img)
            c = _OVERLAY_COLORS_RGB[colour_idx % len(_OVERLAY_COLORS_RGB)]
            draw.line(pixel_pts, fill=c, width=2)
            canvas = np.array(overlay_img)

        colour_idx += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(str(output_path))

    elapsed = time.perf_counter() - t0
    logger.info("[%s] reconstruction: overlay_comparison done in %.3fs -> %s",
                job_id, elapsed, output_path)
    return output_path


def build_masks(
    curves: Dict[str, Any],
    axis_info: Dict[str, Any],
    original_image_path: Path,
    *,
    has_mapping: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build original-series mask and reconstructed-series mask.

    The two masks are built from DIFFERENT data sources:
      - original_mask   : colour-extracted pixels from the original image
      - reconstructed_mask : fitted polynomial polylines rendered to pixel space

    Stroke width of the reconstructed mask is estimated from the original
    mask's distance transform so both masks have comparable thickness.

    Returns (original_mask, reconstructed_mask) as boolean arrays of shape (H, W).
    """
    img = Image.open(str(original_image_path)).convert("RGB")
    width, height = img.size
    img_array = np.array(img)

    original_mask = _build_original_mask(img_array, curves)

    # Estimate stroke width from original mask
    est_thickness = _estimate_stroke_width(original_mask)

    reconstructed_mask = _build_reconstructed_mask(
        curves, axis_info, width, height, has_mapping, thickness=est_thickness,
    )

    # Sanity: masks must be independent arrays
    assert original_mask is not reconstructed_mask, "Masks must not alias"

    orig_nz = int(original_mask.sum())
    recon_nz = int(reconstructed_mask.sum())
    logger.info("build_masks: original_nonzero=%d  reconstructed_nonzero=%d  "
                "shape=%s  est_stroke_width=%d",
                orig_nz, recon_nz, original_mask.shape, est_thickness)
    if orig_nz == 0:
        logger.warning("build_masks: original mask is EMPTY – delta will be unreliable")
    if recon_nz == 0:
        logger.warning("build_masks: reconstructed mask is EMPTY – delta will be unreliable")

    return original_mask, reconstructed_mask


def _estimate_stroke_width(mask: np.ndarray) -> int:
    """Estimate average stroke width from a binary mask using distance transform.

    For thin series lines this typically returns 2-4 px.
    Falls back to 2 if mask is empty.
    """
    if not mask.any():
        return 2
    try:
        from scipy.ndimage import distance_transform_edt
        dt = distance_transform_edt(mask)
        # Skeleton-like: pixels where DT is locally maximal along stroke
        # Use median of nonzero DT values as half-width estimate
        dt_vals = dt[mask]
        if len(dt_vals) == 0:
            return 2
        half_width = float(np.median(dt_vals))
        thickness = max(2, int(round(half_width * 2)))
        thickness = min(thickness, 8)  # cap to avoid bloat
        logger.debug("_estimate_stroke_width: median_half=%.2f -> thickness=%d", half_width, thickness)
        return thickness
    except Exception:
        return 2


# ------------------------------------------------------------------
# Private helpers – data extraction for PLOTTING
# ------------------------------------------------------------------

def _get_series_xy(
    cdata: Dict[str, Any],
    has_mapping: bool,
    img_width: int,
    img_height: int,
) -> Tuple[List[float], List[float]]:
    """Return (xs, ys) for matplotlib plotting.

    Priority: fitted_points (smooth polynomial) > axis_coords > pixel_coords.
    The polynomial fit produces the cleanest, most readable reconstruction.
    """
    # 1) fitted_points — smooth polynomial, best for readable plot
    fit = cdata.get("fit_result") or {}
    fitted_pts = fit.get("fitted_points", []) if isinstance(fit, dict) else []
    if fitted_pts:
        logger.debug("_get_series_xy: using fitted_points (%d pts)", len(fitted_pts))
        return ([p["x"] for p in fitted_pts], [p["y"] for p in fitted_pts])

    # 2) axis_coords — actual data-space coordinates
    axis_coords = cdata.get("axis_coords") or []
    if axis_coords:
        sorted_ac = sorted(axis_coords, key=lambda p: p[0])
        logger.debug("_get_series_xy: using axis_coords (%d pts)", len(sorted_ac))
        return ([p[0] for p in sorted_ac], [p[1] for p in sorted_ac])

    # 3) pixel_coords as last resort
    pixel_coords = cdata.get("pixel_coords") or []
    if pixel_coords:
        sorted_px = sorted(pixel_coords, key=lambda p: p[0])
        logger.debug("_get_series_xy: using pixel_coords (%d pts)", len(sorted_px))
        return ([p[0] for p in sorted_px], [p[1] for p in sorted_px])

    return ([], [])


# ------------------------------------------------------------------
# Private helpers – RECONSTRUCTED points for overlay & mask
# ------------------------------------------------------------------

def _get_reconstructed_pixel_points(
    cdata: Dict[str, Any],
    axis_info: Dict[str, Any],
    img_width: int,
    img_height: int,
    has_mapping: bool,
) -> List[Tuple[int, int]]:
    """Get reconstructed points in pixel space for overlay and mask building.

    Uses the fitted/processed representation of the curve (NOT raw extracted
    pixels) to ensure the overlay shows the *reconstruction's* interpretation
    of the curve rather than re-drawing the original.

    Priority: fitted_points → pixel > axis_coords → pixel > pixel_coords.
    """
    # Extract plot_area stored in the curve dict (if available)
    pa = cdata.get("plot_area")  # [left, top, right, bottom] or None

    # 1) fitted_points → convert from data space to pixel space
    fit = cdata.get("fit_result") or {}
    fitted_pts = fit.get("fitted_points", []) if isinstance(fit, dict) else []
    if fitted_pts and has_mapping:
        coords = [[p["x"], p["y"]] for p in fitted_pts]
        pts = _data_to_pixel_simple(coords, axis_info, img_width, img_height, pa)
        if pts:
            logger.debug("_get_reconstructed_pixel_points: fitted_points->pixel (%d pts)", len(pts))
            return pts

    # 2) axis_coords → convert back to pixel
    axis_coords = cdata.get("axis_coords") or []
    if axis_coords and has_mapping:
        pts = _data_to_pixel_simple(axis_coords, axis_info, img_width, img_height, pa)
        if pts:
            logger.debug("_get_reconstructed_pixel_points: axis_coords->pixel (%d pts)", len(pts))
            return pts

    # 3) pixel_coords as last resort (only used when no mapping)
    pixel_coords = cdata.get("pixel_coords") or []
    if pixel_coords:
        pts = [(int(p[0]), int(p[1])) for p in pixel_coords]
        logger.debug("_get_reconstructed_pixel_points: pixel_coords fallback (%d pts)", len(pts))
        return _sort_points(pts)

    return []


def _data_to_pixel_simple(
    data_coords: List,
    axis_info: Dict[str, Any],
    img_width: int,
    img_height: int,
    plot_area: Optional[List] = None,
) -> List[Tuple[int, int]]:
    """Simple data → pixel conversion matching CurveDigitizer.normalize_to_axis inverse.

    When *plot_area* ``[left, top, right, bottom]`` is supplied the mapping
    is relative to that sub-region of the full image, consistent with how
    ``CurveDigitizer.normalize_to_axis`` converts pixels → data.
    """
    xMin = float(axis_info.get("xMin", 0) or 0)
    xMax = float(axis_info.get("xMax", 100) or 100)
    yMin = float(axis_info.get("yMin", 0) or 0)
    yMax = float(axis_info.get("yMax", 100) or 100)

    dx = xMax - xMin if xMax != xMin else 1.0
    dy = yMax - yMin if yMax != yMin else 1.0

    # Use plot_area if available, else full image
    if plot_area and len(plot_area) == 4:
        p_left, p_top, p_right, p_bottom = (
            int(plot_area[0]), int(plot_area[1]),
            int(plot_area[2]), int(plot_area[3]),
        )
        p_width = p_right - p_left
        p_height = p_bottom - p_top
    else:
        p_left, p_top = 0, 0
        p_width = img_width
        p_height = img_height

    points: List[Tuple[int, int]] = []
    for p in data_coords:
        data_x = p[0] if isinstance(p, (list, tuple)) else p.get("x", 0)
        data_y = p[1] if isinstance(p, (list, tuple)) else p.get("y", 0)
        norm_x = (data_x - xMin) / dx
        norm_y = (data_y - yMin) / dy
        px = int(round(norm_x * p_width + p_left))
        py = int(round((1.0 - norm_y) * p_height + p_top))
        px = max(0, min(px, img_width - 1))
        py = max(0, min(py, img_height - 1))
        points.append((px, py))

    return _sort_points(points)


def _sort_points(pts: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return sorted(pts, key=lambda p: p[0])


# ------------------------------------------------------------------
# Private helpers – mask building
# ------------------------------------------------------------------

def _build_original_mask(img_array: np.ndarray, curves: Dict[str, Any]) -> np.ndarray:
    """Build boolean mask from the original image for extracted curve colors.

    When curves were extracted with the grayscale spatial method their
    ``raw_pixel_points`` are used directly.  Otherwise colour-based
    re-extraction is attempted.

    Suppresses:
      - border/axis bands (outermost pixels)
      - very bright / white background
    """
    from core.image_processor import CurveDigitizer

    h, w = img_array.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    image = Image.fromarray(img_array)

    # Check whether ANY curve was extracted with the grayscale pipeline.
    any_grayscale = any(
        isinstance(cd, dict) and cd.get("extraction_mode") == "grayscale"
        for cd in curves.values()
    )

    if any_grayscale:
        # ── grayscale mode: use stored pixel coordinates directly ──
        for name, cdata in curves.items():
            if not isinstance(cdata, dict):
                continue
            for px, py in (cdata.get("raw_pixel_points") or []):
                if 0 <= py < h and 0 <= px < w:
                    mask[py, px] = True
    else:
        # ── colour mode: re-extract pixels by named colour ──
        dummy_axis = {"xMin": 0, "xMax": 100, "yMin": 0, "yMax": 100}
        digitizer = CurveDigitizer(dummy_axis)

        _DARK_NAMES = {"black", "gray", "grey", "dark gray", "dark grey"}
        curve_colors = [
            (cdata.get("color", name) if isinstance(cdata, dict) else name).lower()
            for name, cdata in curves.items()
        ]
        has_colored_series = any(c not in _DARK_NAMES for c in curve_colors)

        for name, cdata in curves.items():
            if not isinstance(cdata, dict):
                continue
            color = cdata.get("color", name)
            if color.lower() in _DARK_NAMES and has_colored_series:
                logger.debug("_build_original_mask: skipping color '%s'", color)
                continue
            try:
                pixels = digitizer.extract_color_pixels(image, color)
                for px, py in pixels:
                    if 0 <= py < h and 0 <= px < w:
                        mask[py, px] = True
            except Exception:
                pass

        # fallback when no pixels matched
        if not mask.any():
            brightness = img_array.mean(axis=2)
            if has_colored_series:
                r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
                mask = (brightness < 240) & (brightness > 40)
                mx = np.maximum(np.maximum(r, g), b).astype(float)
                mn = np.minimum(np.minimum(r, g), b).astype(float)
                sat = np.where(mx > 0, (mx - mn) / mx, 0)
                mask = mask & (sat >= 0.15)
            else:
                mask = (brightness < 200) & (brightness > 5)

    # Suppress border bands (likely axes/ticks) – 6px margin
    axis_band = 6
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:axis_band, :] = True
    border_mask[-axis_band:, :] = True
    border_mask[:, :axis_band] = True
    border_mask[:, -axis_band:] = True
    mask = mask & ~border_mask

    # Morphological close to connect broken strokes
    try:
        from scipy.ndimage import binary_closing
        struct = np.ones((3, 3), dtype=bool)
        mask = binary_closing(mask, structure=struct, iterations=1)
    except Exception:
        pass

    orig_nz = int(mask.sum())
    logger.info("_build_original_mask: nonzero=%d  shape=%s  (border suppressed, "
                "bw_mode=%s)", orig_nz, mask.shape, not has_colored_series)
    return mask


def _build_reconstructed_mask(
    curves: Dict[str, Any],
    axis_info: Dict[str, Any],
    img_width: int,
    img_height: int,
    has_mapping: bool,
    thickness: int = 2,
) -> np.ndarray:
    """Render RECONSTRUCTED curves (fitted polynomial) into a boolean mask.

    Uses fitted_points → pixel conversion so the mask represents the
    pipeline's reconstruction, NOT the raw extracted pixels.
    Thickness is estimated from the original mask for fair comparison.
    """
    mask = np.zeros((img_height, img_width), dtype=bool)

    # Grayscale-extracted curves don't need colour-based skipping.
    any_grayscale = any(
        isinstance(cd, dict) and cd.get("extraction_mode") == "grayscale"
        for cd in curves.values()
    )

    if not any_grayscale:
        _DARK_NAMES = {"black", "gray", "grey", "dark gray", "dark grey"}
        all_colors = [
            ((cd.get("color", n) if isinstance(cd, dict) else n)).lower()
            for n, cd in curves.items()
            if isinstance(cd, dict) and not cd.get("error")
        ]
        has_colored = any(c not in _DARK_NAMES for c in all_colors)
    else:
        has_colored = False  # irrelevant — nothing will be skipped

    for name, cdata in curves.items():
        if not isinstance(cdata, dict) or cdata.get("error"):
            continue
        if not any_grayscale:
            color = cdata.get("color", name)
            if isinstance(color, str) and color.lower() in _DARK_NAMES and has_colored:
                continue
        pts = _get_reconstructed_pixel_points(cdata, axis_info, img_width, img_height, has_mapping)
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            _draw_line_on_mask(mask, x0, y0, x1, y1, thickness=thickness)

    return mask


def _draw_line_on_mask(
    mask: np.ndarray,
    x0: int, y0: int,
    x1: int, y1: int,
    thickness: int = 2,
) -> None:
    """Bresenham-ish thick line rasteriser."""
    h, w = mask.shape
    n = max(abs(x1 - x0), abs(y1 - y0), 1)
    for t in range(n + 1):
        frac = t / n
        x = int(x0 + frac * (x1 - x0))
        y = int(y0 + frac * (y1 - y0))
        for dy in range(-thickness // 2, thickness // 2 + 1):
            for dx in range(-thickness // 2, thickness // 2 + 1):
                yy = y + dy
                xx = x + dx
                if 0 <= yy < h and 0 <= xx < w:
                    mask[yy, xx] = True
