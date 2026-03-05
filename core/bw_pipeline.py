"""
Advanced B/W curve extraction pipeline.

Implements skeleton-based curve extraction with:
  - Plot-area detection and text/label removal
  - Adaptive binarization + morphological skeleton
  - Dashed/dotted line rejection
  - Anchor-guided A* curve tracing
  - Curve endpoint extension
  - Savitzky-Golay smoothing

This module is ONLY used for B/W images.  The colour pipeline is
completely unaffected.
"""

from __future__ import annotations

import heapq
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_opening,
    distance_transform_edt,
    grey_closing,
    label as ndimage_label,
    uniform_filter,
)

from core.config import BWPipelineConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

_DEBUG_DIR = os.environ.get("CURVE_DEBUG_IMAGES", "")


def _save_debug(tag: str, array: np.ndarray) -> None:
    if not _DEBUG_DIR:
        return
    try:
        out = Path(_DEBUG_DIR)
        out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array.astype(np.uint8) if array.dtype == bool else array).save(
            str(out / f"bw_{tag}.png")
        )
    except Exception:
        pass


# ====================================================================
# 1. Plot-area detection (robust)
# ====================================================================

def detect_plot_area_robust(
    image: Image.Image,
    *,
    dark_threshold: int = 80,
    line_ratio: float = 0.25,
    padding_frac: float = 0.02,
) -> Tuple[int, int, int, int]:
    """Detect plot-area bounding box using axis-line detection.

    Uses HoughLinesP-style detection: find long dark horizontal/vertical
    segments, cluster them, and infer the plot rectangle.

    Returns (left, top, right, bottom) in pixel coords.
    """
    img_array = np.array(image.convert("RGB"))
    height, width = img_array.shape[:2]

    gray = np.mean(img_array[:, :, :3], axis=2)
    dark = gray < dark_threshold

    # --- Horizontal axis lines (rows) ---
    dark_per_row = np.sum(dark, axis=1)
    min_dark_h = int(width * line_ratio)
    h_rows = np.where(dark_per_row >= min_dark_h)[0]

    # --- Vertical axis lines (cols) ---
    dark_per_col = np.sum(dark, axis=0)
    min_dark_v = int(height * line_ratio)
    v_cols = np.where(dark_per_col >= min_dark_v)[0]

    # Filter: only solid lines (>70% fill)
    def _is_solid_row(r):
        idx = np.where(dark[r, :])[0]
        if len(idx) < 5:
            return False
        return len(idx) / (idx[-1] - idx[0] + 1) > 0.70

    def _is_solid_col(c):
        idx = np.where(dark[:, c])[0]
        if len(idx) < 5:
            return False
        return len(idx) / (idx[-1] - idx[0] + 1) > 0.70

    h_rows = np.array([r for r in h_rows if _is_solid_row(r)], dtype=int)
    v_cols = np.array([c for c in v_cols if _is_solid_col(c)], dtype=int)

    # Defaults
    plot_left = int(width * 0.10)
    plot_right = int(width * 0.90)
    plot_top = int(height * 0.05)
    plot_bottom = int(height * 0.85)

    if len(v_cols) > 0:
        left_cands = v_cols[v_cols < width // 2]
        right_cands = v_cols[v_cols > width // 2]
        if len(left_cands) > 0:
            plot_left = int(np.max(left_cands))
        if len(right_cands) > 0:
            plot_right = int(np.min(right_cands))

    if len(h_rows) > 0:
        bottom_cands = h_rows[h_rows > height // 2]
        top_cands = h_rows[h_rows < height // 2]
        if len(bottom_cands) > 0:
            plot_bottom = int(np.min(bottom_cands))
        if len(top_cands) > 0:
            plot_top = int(np.max(top_cands))

    # Sanity
    if plot_right - plot_left < width * 0.2:
        plot_left = int(width * 0.10)
        plot_right = int(width * 0.90)
    if plot_bottom - plot_top < height * 0.2:
        plot_top = int(height * 0.05)
        plot_bottom = int(height * 0.85)

    # Add small inward padding to skip axis-line pixels
    pad_x = int((plot_right - plot_left) * padding_frac)
    pad_y = int((plot_bottom - plot_top) * padding_frac)
    plot_left += pad_x
    plot_right -= pad_x
    plot_top += pad_y
    plot_bottom -= pad_y

    logger.debug("detect_plot_area_robust: (%d, %d, %d, %d)",
                 plot_left, plot_top, plot_right, plot_bottom)
    return (plot_left, plot_top, plot_right, plot_bottom)


# ====================================================================
# 2. Preprocessing: binarize, remove text, skeletonize
# ====================================================================

def preprocess_bw(
    image: Image.Image,
    plot_area: Tuple[int, int, int, int],
    *,
    text_area_max_ratio: float = 0.008,
    text_aspect_min: float = 0.2,
    text_aspect_max: float = 5.0,
    close_kernel_size: Tuple[int, int] = (3, 7),
    config: Optional[BWPipelineConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """Preprocess a B/W image for skeleton extraction.

    Steps:
      1. Crop to plot area
      2. Adaptive threshold (Otsu+margin)
      3. Remove text/labels via connected-component analysis
      4. Morphological closing to bridge gaps
      5. Remove axis remnants (long straight lines at borders)
      6. Skeletonize (Zhang-Suen)

    Returns (skeleton, binary_cleaned, adjusted_plot_area).
         skeleton: bool array, shape (crop_h, crop_w)
         binary_cleaned: bool array before skeletonize
         adjusted_plot_area: the actual (left, top, right, bottom) used
    """
    cfg = config or DEFAULT_CONFIG

    img_array = np.array(image.convert("RGB"))
    p_left, p_top, p_right, p_bottom = plot_area

    # Crop to plot area
    region = img_array[p_top:p_bottom, p_left:p_right]
    if region.ndim == 3:
        gray = np.mean(region[:, :, :3].astype(np.float32), axis=2).astype(np.uint8)
    else:
        gray = region.astype(np.uint8)
    rh, rw = gray.shape

    # ── Enhanced preprocessing: CLAHE contrast normalization ──
    if cfg.clahe_clip_limit > 0:
        gray = _apply_clahe(gray, cfg.clahe_clip_limit, cfg.clahe_tile_grid_size)
        _save_debug("preprocess_clahe", gray)

    # Denoise (median filter for general use)
    gray = np.array(Image.fromarray(gray).filter(ImageFilter.MedianFilter(size=3)))

    # ── Enhanced preprocessing: Black-hat stroke enhancement ──
    if cfg.use_blackhat:
        gray = _apply_blackhat(gray, cfg.blackhat_kernel)
        _save_debug("preprocess_blackhat", gray)

    # Adaptive Otsu threshold
    threshold = _otsu_threshold(gray)
    binary = gray <= min(threshold + 10, 200)

    # ── Enhanced preprocessing: density check + adaptive fallback ──
    if cfg.adaptive_thresh:
        fg_ratio = float(binary.sum()) / max(binary.size, 1)
        if fg_ratio > 0.50 or fg_ratio < 0.01:
            logger.info("preprocess_bw: Otsu fg_ratio=%.3f out of range, "
                        "falling back to adaptive threshold", fg_ratio)
            binary = _adaptive_threshold(gray, cfg.adaptive_block_size,
                                         cfg.adaptive_C)

    _save_debug("preprocess_binary", binary.astype(np.uint8) * 255)

    # ── Enhanced preprocessing: Hough-based axis/gridline removal ──
    if cfg.hough_remove_axes:
        binary = _remove_lines_hough(binary, rh, rw, cfg)
        _save_debug("preprocess_hough_cleaned", binary.astype(np.uint8) * 255)

    # ── Correct operation order: grid → text → tick → close → noise → border → skel ──
    # This order ensures:
    #   - Gridlines removed before they can be merged with curves by closing
    #   - Text removed before closing can merge chars with adjacent curves
    #   - Closing bridges gaps in CLEAN curve-only pixels
    #   - Text removal is deterministic in ONE pass (no need to run twice)

    # 1. Remove gridlines (morphological line kernels)
    binary = _remove_gridlines_morph(binary, rh, rw)
    _save_debug("preprocess_no_grid", binary.astype(np.uint8) * 255)

    # 2. Remove text BEFORE closing (prevents text merging with curves)
    binary = _remove_text_components(binary, rh, rw,
                                     area_max_ratio=text_area_max_ratio,
                                     aspect_min=text_aspect_min,
                                     aspect_max=text_aspect_max,
                                     min_curve_span_ratio=0.15)
    _save_debug("preprocess_no_text", binary.astype(np.uint8) * 255)

    # 3. Remove tick marks near axes
    binary = _remove_tick_marks(binary, rh, rw)

    # 4. Morphological close AFTER cleaning (bridge gaps in curves only)
    ck = np.ones(close_kernel_size, dtype=bool)
    binary = binary_closing(binary, structure=ck, iterations=1)

    # 5. Remove isolated single-pixel noise
    padded = np.pad(binary.astype(np.uint8), 1, mode='constant')
    neighbor_count = np.zeros((rh, rw), dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbor_count += padded[1 + dy:rh + 1 + dy,
                                     1 + dx:rw + 1 + dx]
    binary[binary & (neighbor_count == 0)] = False

    # 6. Remove axis remnants at borders (reduced to 3px to avoid
    #    clipping curve endpoints near the axis)
    binary = _remove_border_lines(binary, border_px=3)

    # 7. Skeletonize
    skeleton = _skeletonize(binary)
    _save_debug("preprocess_skeleton", skeleton.astype(np.uint8) * 255)

    return skeleton, binary, (p_left, p_top, p_right, p_bottom)


def _remove_text_components(
    binary: np.ndarray,
    img_h: int,
    img_w: int,
    *,
    area_max_ratio: float = 0.008,
    aspect_min: float = 0.2,
    aspect_max: float = 5.0,
    min_curve_span_ratio: float = 0.15,
    euler_threshold: int = -1,
) -> np.ndarray:
    """Remove connected components that are likely text / numeric labels.

    Robust text detection heuristics (catches labels like '110%', '80',
    '%', 'SPEED' that appear inside the plot area):

      1. **Size gate**: components smaller than ``area_max_ratio`` of the
         plot area are candidates (curves are normally much larger).
      2. **Aspect ratio**: character bounding boxes are roughly square or
         slightly tall/wide (0.2–5.0).
      3. **Euler number**: text characters and digit strings tend to have
         holes (e.g., '0', '8', '%') giving Euler number ≤ 0.  Real
         curves are simple arcs with Euler number = 1.
      4. **Density / compactness**: text fills its bounding box densely
         (> 15 %).  A thin curve crossing the same area would fill < 10 %.
      5. **Horizontal span**: real curves span a large fraction of the
         plot width; text spans < ``min_curve_span_ratio``.
      6. **Branch density**: text has many branch-points per unit length
         (junctions in letters like 'E', '%', 'S').

    Components that fail ALL curve-like checks and pass text-like checks
    are removed.  Large components and wide-spanning components are
    always kept.
    """
    structure = np.ones((3, 3), dtype=int)
    labelled, n_comp = ndimage_label(binary, structure=structure)

    if n_comp == 0:
        return binary

    plot_area_px = img_h * img_w
    area_max = int(plot_area_px * area_max_ratio)
    min_curve_span = int(img_w * min_curve_span_ratio)

    result = binary.copy()
    for comp_id in range(1, n_comp + 1):
        comp_mask = labelled == comp_id
        area = int(comp_mask.sum())

        # Tiny speck — always remove
        if area < 4:
            result[comp_mask] = False
            continue

        # Large component — always keep (likely a curve)
        if area > area_max:
            continue

        # Bounding box
        ys, xs = np.where(comp_mask)
        bbox_h = int(ys.max() - ys.min()) + 1
        bbox_w = int(xs.max() - xs.min()) + 1
        aspect = bbox_w / max(bbox_h, 1)
        density = area / max(bbox_h * bbox_w, 1)

        # Wide-spanning component — likely a curve, keep it
        if bbox_w >= min_curve_span:
            continue

        # --- Text scoring (accumulate evidence) ---
        text_evidence = 0.0

        # (a) Aspect ratio typical of characters / short labels
        if aspect_min <= aspect <= aspect_max:
            text_evidence += 0.15

        # (b) Dense fill (text chars fill their bbox more than thin curves)
        if density > 0.15:
            text_evidence += 0.15
        if density > 0.30:
            text_evidence += 0.10

        # (c) Euler number: count holes via flood-fill of bg inside bbox
        euler = _euler_number_component(comp_mask, ys, xs)
        if euler <= euler_threshold:
            # Has holes (digits 0, 4, 6, 8, 9; '%' sign)
            text_evidence += 0.25

        # (d) Short horizontal span: curves are wide, text is narrow
        span_ratio = bbox_w / max(img_w, 1)
        if span_ratio < 0.08:
            text_evidence += 0.20
        elif span_ratio < min_curve_span_ratio:
            text_evidence += 0.10

        # (e) Branch density: text has many junctions per pixel
        n_branches = _fast_branch_count(comp_mask, ys, xs)
        branch_density = n_branches / max(area, 1)
        if branch_density > 0.02:
            text_evidence += 0.15
        elif branch_density > 0.01:
            text_evidence += 0.08

        # (f) Near margins: labels often appear near axes or edges
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        margin_x = img_w * 0.12
        margin_y = img_h * 0.12
        near_margin = (cx < margin_x or cx > img_w - margin_x or
                       cy < margin_y or cy > img_h - margin_y)
        if near_margin:
            text_evidence += 0.10

        # --- Decision ---
        if text_evidence >= 0.40:
            result[comp_mask] = False

    return result


def _euler_number_component(
    comp_mask: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
) -> int:
    """Estimate the Euler number (1 - #holes) of a binary component.

    Uses the bounding-box sub-region to count internal background
    connected components (holes).
    """
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sub = comp_mask[y0:y1, x0:x1]

    # Pad with False so bg around the component is connected
    padded = np.pad(sub, 1, mode='constant', constant_values=False)
    bg = ~padded
    bg_labelled, n_bg = ndimage_label(bg, structure=np.ones((3, 3), dtype=int))
    # One bg region is the outer background; rest are holes
    n_holes = max(n_bg - 1, 0)
    return 1 - n_holes


def _fast_branch_count(
    comp_mask: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
) -> int:
    """Count branch-points (pixels with ≥ 3 neighbours) in a component.

    Operates only on the bounding-box sub-region for speed.
    """
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sub = comp_mask[y0:y1, x0:x1].astype(np.uint8)
    padded = np.pad(sub, 1, mode='constant')
    nbr = np.zeros_like(sub, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nbr += padded[1 + dy:padded.shape[0] - 1 + dy,
                          1 + dx:padded.shape[1] - 1 + dx]
    return int(np.sum((sub > 0) & (nbr >= 3)))


def _remove_border_lines(binary: np.ndarray, border_px: int = 5) -> np.ndarray:
    """Remove dark pixels within border_px of the image edges."""
    result = binary.copy()
    result[:border_px, :] = False
    result[-border_px:, :] = False
    result[:, :border_px] = False
    result[:, -border_px:] = False
    return result


# ====================================================================
# 2b. Robust gridline removal (morphological)
# ====================================================================

def _remove_gridlines_morph(
    binary: np.ndarray,
    img_h: int,
    img_w: int,
    *,
    h_kernel_ratio: float = 0.20,
    v_kernel_ratio: float = 0.20,
    fill_threshold: float = 0.80,
    protect_curves: bool = True,
) -> np.ndarray:
    """Remove gridlines using morphological opening with line kernels.

    Strategy:
      1. Detect horizontal lines via opening with (1, K_h) kernel.
      2. Detect vertical lines via opening with (K_v, 1) kernel.
      3. Create grid mask from detected lines.
      4. If *protect_curves*: keep pixels where curves cross gridlines
         (prevents 1-pixel gaps in curves at grid crossings).
      5. Subtract grid mask from binary.
      6. Fallback: also clear any rows/cols with >80 % fill.

    This is superior to the simple fill-ratio approach because it
    targets actual long straight lines rather than clearing entire
    rows/columns (which damages curves that happen to run
    horizontally).

    Parameters
    ----------
    binary : np.ndarray (bool)
        Binarized plot-area image.
    img_h, img_w : int
        Dimensions of the binary image.
    h_kernel_ratio : float
        Horizontal kernel length as fraction of image width.
    v_kernel_ratio : float
        Vertical kernel length as fraction of image height.
    fill_threshold : float
        Fallback: clear rows/cols with fill ratio above this.
    protect_curves : bool
        If True, preserve curve pixels at grid crossings.

    Returns
    -------
    np.ndarray (bool)
        Binary with gridlines removed.
    """
    result = binary.copy()

    # Minimum kernel length for reliable line detection
    h_klen = max(25, int(img_w * h_kernel_ratio))
    v_klen = max(25, int(img_h * v_kernel_ratio))

    # Detect horizontal grid lines
    h_kernel = np.ones((1, h_klen), dtype=bool)
    h_lines = binary_opening(binary, structure=h_kernel, iterations=1)

    # Detect vertical grid lines
    v_kernel = np.ones((v_klen, 1), dtype=bool)
    v_lines = binary_opening(binary, structure=v_kernel, iterations=1)

    # Filter: only keep detections that span ≥ 85 % of full width/height.
    # This prevents shallow curves from being misidentified as grid lines.
    SPAN_RATIO = 0.85
    if h_lines.any():
        for r in range(img_h):
            row_d = h_lines[r, :]
            if not row_d.any():
                continue
            c = np.where(row_d)[0]
            if (c[-1] - c[0] + 1) < img_w * SPAN_RATIO:
                h_lines[r, :] = False

    if v_lines.any():
        for c in range(img_w):
            col_d = v_lines[:, c]
            if not col_d.any():
                continue
            rows = np.where(col_d)[0]
            if (rows[-1] - rows[0] + 1) < img_h * SPAN_RATIO:
                v_lines[:, c] = False

    grid_mask = h_lines | v_lines

    if protect_curves and grid_mask.any():
        # Find non-grid foreground pixels
        non_grid = binary & ~grid_mask
        if non_grid.any():
            # Dilate non-grid pixels to mark "curve vicinity"
            struct3 = np.ones((3, 3), dtype=bool)
            curve_vicinity = binary_dilation(
                non_grid, structure=struct3, iterations=1,
            )
            # Protect grid pixels that sit in curve vicinity
            # (these are curve–grid crossing points)
            protection = grid_mask & curve_vicinity
            grid_mask = grid_mask & ~protection

    result[grid_mask] = False
    _save_debug("gridlines_morph_removed", result.astype(np.uint8) * 255)

    # Fallback: also catch lighter grids via fill-ratio
    row_fill = result.sum(axis=1) / max(img_w, 1)
    result[row_fill > fill_threshold, :] = False
    col_fill = result.sum(axis=0) / max(img_h, 1)
    result[:, col_fill > fill_threshold] = False

    return result


# ====================================================================
# 2c. Tick mark removal
# ====================================================================

def _remove_tick_marks(
    binary: np.ndarray,
    img_h: int,
    img_w: int,
    *,
    max_length: int = 20,
    max_thickness: int = 5,
    proximity: int = 12,
) -> np.ndarray:
    """Remove tick marks near axes (short perpendicular lines at edges).

    Tick marks are small line segments near the borders of the plot area
    that are perpendicular to the axes.  They typically appear at the
    left/bottom edges where axis ticks protrude into the plot.

    Only removes small components whose bounding box fits within
    *max_length* × *max_thickness* and that touch an edge within
    *proximity* pixels.  This is safe for curves because real curves
    are much larger and span far from the edges.

    Parameters
    ----------
    binary : np.ndarray (bool)
    img_h, img_w : int
    max_length : int
        Maximum extent of a tick mark in its long dimension.
    max_thickness : int
        Maximum extent in the perpendicular dimension.
    proximity : int
        Maximum distance from the plot edge to qualify as a tick.

    Returns
    -------
    np.ndarray (bool)
    """
    structure = np.ones((3, 3), dtype=int)
    labelled, n_comp = ndimage_label(binary, structure=structure)

    if n_comp == 0:
        return binary

    result = binary.copy()

    for comp_id in range(1, n_comp + 1):
        comp_mask = labelled == comp_id
        ys, xs = np.where(comp_mask)
        if len(ys) < 2:
            continue

        bbox_h = int(ys.max() - ys.min()) + 1
        bbox_w = int(xs.max() - xs.min()) + 1

        # Check proximity to edges
        near_left = int(xs.min()) < proximity
        near_right = int(xs.max()) > img_w - proximity
        near_top = int(ys.min()) < proximity
        near_bottom = int(ys.max()) > img_h - proximity

        if not (near_left or near_right or near_top or near_bottom):
            continue

        # Tick at left/right edge: short horizontal, thin vertical
        if (near_left or near_right):
            if bbox_w <= max_length and bbox_h <= max_thickness:
                result[comp_mask] = False
                continue

        # Tick at top/bottom edge: short vertical, thin horizontal
        if (near_top or near_bottom):
            if bbox_h <= max_length and bbox_w <= max_thickness:
                result[comp_mask] = False
                continue

        # Small blob very close to edge
        area = int(comp_mask.sum())
        if area <= max_length * max_thickness:
            if bbox_w <= max_length and bbox_h <= max_length:
                result[comp_mask] = False

    return result


def _skeletonize(binary: np.ndarray) -> np.ndarray:
    """Thin binary image to 1-pixel skeleton."""
    try:
        from skimage.morphology import skeletonize as sk_skeletonize
        return sk_skeletonize(binary).astype(bool)
    except ImportError:
        # Fallback: iterative morphological thinning
        return _morphological_thin(binary)


def _morphological_thin(binary: np.ndarray) -> np.ndarray:
    """Simple morphological thinning fallback."""
    from scipy.ndimage import binary_erosion
    skel = np.zeros_like(binary)
    temp = binary.copy()
    element = np.ones((3, 3), dtype=bool)
    while True:
        eroded = binary_erosion(temp, structure=element)
        opened = binary_closing(eroded, structure=element)
        subset = temp & ~opened
        skel |= subset
        if not eroded.any():
            break
        temp = eroded
    return skel


def _otsu_threshold(gray: np.ndarray) -> int:
    """Compute Otsu threshold (deterministic)."""
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    total = gray.size
    sum_total = float(np.dot(np.arange(256), hist))
    sum_bg = 0.0
    weight_bg = 0
    max_var = 0.0
    best_t = 128
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > max_var:
            max_var = var_between
            best_t = t
    return max(40, min(best_t, 200))


# ====================================================================
# 2b. Enhanced preprocessing helpers (CLAHE, blackhat, adaptive, Hough)
# ====================================================================


def _apply_clahe(
    gray: np.ndarray, clip_limit: float, tile_size: Tuple[int, int],
) -> np.ndarray:
    """Apply CLAHE contrast normalization.

    Tries OpenCV first (faster), falls back to scikit-image.
    """
    try:
        import cv2  # type: ignore
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        return clahe.apply(gray)
    except Exception:
        pass
    try:
        from skimage.exposure import equalize_adapthist
        gray_f = gray.astype(np.float64) / 255.0
        # skimage clip_limit is on a different scale; convert
        sk_clip = max(clip_limit / 100.0, 0.005)
        enhanced = equalize_adapthist(
            gray_f, clip_limit=sk_clip, kernel_size=tile_size,
        )
        return (enhanced * 255).astype(np.uint8)
    except Exception as exc:
        logger.debug("CLAHE unavailable, skipping: %s", exc)
        return gray


def _apply_blackhat(
    gray: np.ndarray, kernel_size: Tuple[int, int],
) -> np.ndarray:
    """Black-hat transform to highlight dark strokes on light background.

    blackhat = closing(gray) - gray
    """
    closed = grey_closing(gray, size=kernel_size)
    bh = closed.astype(np.int16) - gray.astype(np.int16)
    bh = np.clip(bh, 0, 255).astype(np.uint8)
    # Only use if it reveals meaningful contrast
    if bh.max() > 30:
        return bh
    logger.debug("blackhat max=%d too low, keeping original", bh.max())
    return gray


def _adaptive_threshold(
    gray: np.ndarray, block_size: int, C: int,
) -> np.ndarray:
    """Adaptive threshold using local-mean comparison (no OpenCV needed)."""
    local_mean = uniform_filter(gray.astype(np.float64), size=block_size)
    return gray < (local_mean - C)


def _remove_lines_hough(
    binary: np.ndarray,
    img_h: int,
    img_w: int,
    config: BWPipelineConfig,
) -> np.ndarray:
    """Remove long straight lines (axes/gridlines) via Hough transform."""
    try:
        from skimage.transform import probabilistic_hough_line
    except ImportError:
        logger.debug("skimage Hough unavailable, skipping")
        return binary

    edges = binary.astype(np.uint8)
    min_len = max(config.hough_min_line_length, int(min(img_h, img_w) * 0.3))

    lines = probabilistic_hough_line(
        edges,
        threshold=config.hough_threshold,
        line_length=min_len,
        line_gap=config.hough_max_line_gap,
    )
    if not lines:
        return binary

    result = binary.copy()
    margin = 5  # pixels from border considered "axis"

    for (x0, y0), (x1, y1) in lines:
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        length = math.sqrt(dx * dx + dy * dy)

        # Only remove near-horizontal or near-vertical lines
        if dx > 0 and dy > 0:
            angle = math.atan2(dy, dx)
            if 0.1 < angle < (math.pi / 2 - 0.1):
                continue  # diagonal — skip

        is_removable = False

        if dy <= 3:  # horizontal
            if length > img_w * 0.5:
                y_mid = (y0 + y1) // 2
                if y_mid < margin or y_mid > img_h - margin:
                    is_removable = True
                elif length > img_w * 0.7:
                    is_removable = True
        elif dx <= 3:  # vertical
            if length > img_h * 0.5:
                x_mid = (x0 + x1) // 2
                if x_mid < margin or x_mid > img_w - margin:
                    is_removable = True
                elif length > img_h * 0.7:
                    is_removable = True

        if is_removable:
            _rasterize_line_remove(result, x0, y0, x1, y1, thickness=3)

    logger.debug("Hough removed %d/%d lines",
                 sum(1 for _ in lines if True), len(lines))
    return result


def _rasterize_line_remove(
    binary: np.ndarray, x0: int, y0: int, x1: int, y1: int,
    thickness: int = 3,
) -> None:
    """Erase a line from *binary* in-place using Bresenham rasterisation."""
    h, w = binary.shape
    n = max(abs(x1 - x0), abs(y1 - y0), 1)
    half = thickness // 2
    for t in range(n + 1):
        frac = t / n
        cx = int(round(x0 + frac * (x1 - x0)))
        cy = int(round(y0 + frac * (y1 - y0)))
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w:
                    binary[ny, nx] = False


# ====================================================================
# 3. Dashed/dotted line scoring and rejection
# ====================================================================

def score_dashed(
    component_mask: np.ndarray,
    skeleton_pixels: List[Tuple[int, int]],
) -> float:
    """Score a skeleton component for dashed/dotted characteristics.

    Returns a score in [0, 1] where higher = more likely dashed/dotted.

    Scoring criteria:
      - endpoints_per_length: dashed lines have many endpoints per unit length
      - gap_periodicity: regular gaps indicate dashing pattern
      - fragmentation: many small sub-segments
    """
    if len(skeleton_pixels) < 5:
        return 0.2  # Too few pixels to judge — give benefit of doubt

    arr = np.array(skeleton_pixels)
    xs, ys = arr[:, 0], arr[:, 1]

    # Arc length of the component
    total_length = _arc_length(skeleton_pixels)
    if total_length < 10:
        return 0.2  # Short but not necessarily dashed

    # Count endpoints (pixels with exactly 1 neighbor in skeleton)
    n_endpoints = _count_endpoints(component_mask)
    epl = n_endpoints / max(total_length, 1)

    # A solid line has ~2 endpoints; dashed has many
    # Normalize: solid ≈ 0, heavily dashed ≈ 1
    epl_score = min(1.0, epl * 20)  # empirical scaling

    # Gap analysis: project onto dominant direction and look for gaps
    gap_score = _gap_periodicity_score(skeleton_pixels)

    # Combined score (weighted)
    score = 0.5 * epl_score + 0.5 * gap_score
    return float(min(1.0, max(0.0, score)))


def _arc_length(pixels: List[Tuple[int, int]]) -> float:
    """Approximate arc length of ordered pixel sequence."""
    if len(pixels) < 2:
        return 0.0
    arr = np.array(pixels, dtype=float)
    diffs = np.diff(arr, axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))


def _count_endpoints(mask: np.ndarray) -> int:
    """Count skeleton endpoints (pixels with exactly 1 neighbor)."""
    h, w = mask.shape
    count = 0
    ys, xs = np.where(mask)
    for y, x in zip(ys, xs):
        neighbors = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
                    neighbors += 1
        if neighbors == 1:
            count += 1
    return count


def _gap_periodicity_score(pixels: List[Tuple[int, int]]) -> float:
    """Score gap periodicity by projecting onto dominant axis."""
    if len(pixels) < 10:
        return 0.0

    arr = np.array(pixels, dtype=float)
    xs = arr[:, 0]

    # Project onto x-axis (assuming mostly-horizontal curves)
    x_sorted = np.sort(np.unique(xs.astype(int)))
    if len(x_sorted) < 5:
        return 0.0

    # Find gaps (consecutive x values with no skeleton pixel)
    diffs = np.diff(x_sorted)
    gaps = diffs[diffs > 2]  # gaps larger than 2px

    if len(gaps) < 2:
        return 0.0  # No significant gaps → solid

    # Periodicity: low coefficient of variation of gap sizes → periodic
    gap_mean = float(np.mean(gaps))
    gap_std = float(np.std(gaps))
    cv = gap_std / max(gap_mean, 1e-6)

    # Many gaps with regular spacing → dashed
    gap_density = len(gaps) / max(len(x_sorted), 1)
    periodicity = max(0, 1.0 - cv) * min(1.0, gap_density * 10)

    return float(min(1.0, periodicity))


def filter_dashed_components(
    skeleton: np.ndarray,
    *,
    dashed_threshold: float = 0.45,
    min_span_ratio: float = 0.10,
) -> np.ndarray:
    """Remove skeleton components that score as dashed/dotted.

    Returns cleaned skeleton with dashed components removed.
    """
    h, w = skeleton.shape
    structure = np.ones((3, 3), dtype=int)
    labelled, n_comp = ndimage_label(skeleton, structure=structure)

    if n_comp == 0:
        return skeleton

    result = skeleton.copy()
    min_span = int(w * min_span_ratio)

    for comp_id in range(1, n_comp + 1):
        comp_mask = labelled == comp_id
        ys, xs = np.where(comp_mask)
        if len(xs) == 0:
            continue

        x_span = int(xs.max() - xs.min())

        # Small fragments: remove if tiny
        if x_span < min_span and len(xs) < 20:
            result[comp_mask] = False
            continue

        pixels = list(zip(xs.tolist(), ys.tolist()))
        score = score_dashed(comp_mask, pixels)

        if score > dashed_threshold:
            logger.debug("filter_dashed: removing component %d (score=%.3f, "
                        "span=%d, n_px=%d)", comp_id, score, x_span, len(xs))
            result[comp_mask] = False

    _save_debug("dashed_filtered", result.astype(np.uint8) * 255)
    return result


# ====================================================================
# 4. Connected component extraction from skeleton
# ====================================================================

def extract_skeleton_components(
    skeleton: np.ndarray,
    *,
    min_pixels: int = 15,
    min_span_ratio: float = 0.08,
) -> List[Dict[str, Any]]:
    """Extract connected components from skeleton with rich scoring.

    Returns list of dicts with:
      - 'pixels': List[(x, y)] in region-local coords
      - 'x_span', 'y_span': bounding box extent
      - 'area': number of pixels
      - 'mask': boolean mask for this component
      - 'dashed_score': dashed/dotted likelihood
      - 'text_score': text-like structure likelihood
      - 'branch_density': branchpoints per unit arc length
      - 'endpoint_density': endpoints per unit arc length
      - 'compactness': area / (x_span * y_span)
    """
    h, w = skeleton.shape
    min_span = int(w * min_span_ratio)

    structure = np.ones((3, 3), dtype=int)
    labelled, n_comp = ndimage_label(skeleton, structure=structure)

    components = []
    for comp_id in range(1, n_comp + 1):
        comp_mask = labelled == comp_id
        ys, xs = np.where(comp_mask)
        if len(xs) < min_pixels:
            continue

        x_span = int(xs.max() - xs.min()) + 1
        y_span = int(ys.max() - ys.min()) + 1

        if x_span < min_span:
            continue

        pixels = list(zip(xs.tolist(), ys.tolist()))
        dashed_score = score_dashed(comp_mask, pixels)

        # --- Compute text / branch scoring ---
        n_endpoints = _count_endpoints(comp_mask)
        n_branches = _count_branchpoints(comp_mask)
        arc_len = _arc_length(pixels) if len(pixels) > 1 else float(len(pixels))
        arc_len = max(arc_len, 1.0)

        endpoint_density = n_endpoints / arc_len
        branch_density = n_branches / arc_len
        compactness = len(pixels) / max(x_span * y_span, 1)

        # Text score: high endpoint/branch density + compact + short span
        span_ratio = x_span / max(w, 1)
        text_score = _compute_text_score(
            endpoint_density, branch_density, compactness,
            span_ratio, x_span, y_span, len(pixels),
        )

        components.append({
            "pixels": pixels,
            "x_span": x_span,
            "y_span": y_span,
            "area": len(pixels),
            "mask": comp_mask,
            "dashed_score": dashed_score,
            "text_score": text_score,
            "branch_density": branch_density,
            "endpoint_density": endpoint_density,
            "compactness": compactness,
            "n_endpoints": n_endpoints,
            "n_branches": n_branches,
            "mean_y": float(np.mean(ys)),
            "x_min": int(xs.min()),
            "x_max": int(xs.max()),
        })

    # Sort by area (largest first)
    components.sort(key=lambda c: c["area"], reverse=True)
    return components


def _count_branchpoints(mask: np.ndarray) -> int:
    """Count skeleton branchpoints (pixels with >= 3 neighbors)."""
    h, w = mask.shape
    count = 0
    ys, xs = np.where(mask)
    for y, x in zip(ys, xs):
        neighbors = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
                    neighbors += 1
        if neighbors >= 3:
            count += 1
    return count


def _compute_text_score(
    endpoint_density: float,
    branch_density: float,
    compactness: float,
    span_ratio: float,
    x_span: int,
    y_span: int,
    area: int,
) -> float:
    """Score how likely a component is text rather than a curve.

    Returns 0..1 where higher = more text-like.

    Text heuristics:
      - High endpoint density (many stroke ends)
      - High branch density (junctions in letters)
      - High compactness (letters cluster tightly)
      - Short horizontal span relative to image width
      - Near-square or tall aspect ratio
    """
    score = 0.0

    # Endpoint density: curves have ~0.01/px, text has >0.05/px
    if endpoint_density > 0.08:
        score += 0.3
    elif endpoint_density > 0.04:
        score += 0.15

    # Branch density: curves rarely branch, text does
    if branch_density > 0.03:
        score += 0.3
    elif branch_density > 0.015:
        score += 0.15

    # Short span relative to image → text label
    if span_ratio < 0.15:
        score += 0.2
    elif span_ratio < 0.25:
        score += 0.1

    # Compact bounding box → text cluster
    if compactness > 0.3 and x_span < 100:
        score += 0.1

    # Tall or square aspect → text (curves are wide)
    aspect = x_span / max(y_span, 1)
    if aspect < 1.5 and area < 500:
        score += 0.1

    return min(1.0, score)


def select_best_curves(
    components: List[Dict[str, Any]],
    num_curves: int,
    *,
    dashed_threshold: float = 0.45,
    text_threshold: float = 0.50,
    ignore_dashed: bool = True,
    auto_detect: bool = True,
) -> List[Dict[str, Any]]:
    """Select the best curve components from candidates.

    Selection criteria:
      - Reject text-like components (text_score > text_threshold)
      - Reject dashed components (dashed_score > dashed_threshold)
      - Reject near-vertical components (likely reference lines)
      - Score: x_span * density * (1 - dashed_score) * (1 - text_score)
      - If auto_detect=True, return max(num_curves, auto-detected good count)
    """
    candidates = []
    for comp in components:
        # Reject text-like components
        text_score = comp.get("text_score", 0.0)
        if text_score > text_threshold:
            logger.debug(
                "Rejecting text component: text_score=%.2f, area=%d, x_span=%d",
                text_score, comp["area"], comp["x_span"],
            )
            continue

        # Reject dashed/dotted components
        if ignore_dashed and comp["dashed_score"] > dashed_threshold:
            logger.debug(
                "Rejecting dashed component: dashed_score=%.2f, area=%d",
                comp["dashed_score"], comp["area"],
            )
            continue

        # Score: prefer large span * density, penalize text/dashed
        density = comp["area"] / max(comp["x_span"], 1)
        score = (
            comp["x_span"]
            * density
            * (1.0 - comp["dashed_score"])
            * (1.0 - text_score)
        )

        # Penalize very steep components (likely vertical reference lines)
        if comp["y_span"] > comp["x_span"] * 1.5:
            score *= 0.1

        candidates.append({**comp, "selection_score": score})

    # Sort by score (best first)
    candidates.sort(key=lambda c: c["selection_score"], reverse=True)

    # Determine how many curves to return
    if auto_detect and len(candidates) > num_curves:
        # Use all valid candidates if auto-detect finds more
        effective_num = len(candidates)
        logger.info(
            "Auto-detected %d curves (requested %d)", effective_num, num_curves,
        )
    else:
        effective_num = num_curves

    selected = candidates[:effective_num]

    # Sort selected by mean_y (topmost first, i.e. lowest y in image coords)
    selected.sort(key=lambda c: c["mean_y"])

    return selected


# ====================================================================
# 5. Anchor-guided A* tracing
# ====================================================================

def trace_with_anchors(
    skeleton: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    *,
    snap_radius: int = 20,
    curvature_penalty: float = 2.0,
    gap_bridge_max: int = 10,
    gap_bridge_cost: float = 5.0,
    tangent_window: int = 5,
    distance_transform: Optional[np.ndarray] = None,
) -> Optional[List[Tuple[int, int]]]:
    """Trace the curve between start and end anchors using A*.

    Parameters
    ----------
    skeleton : np.ndarray (bool)
        Skeleton image (1-pixel wide curves).
    start, end : (x, y)
        Anchor coordinates in skeleton-local space.
    snap_radius : int
        Max distance to snap anchors to nearest skeleton pixel.
    curvature_penalty : float
        Cost multiplier for sharp turns.
    gap_bridge_max : int
        Max pixels to bridge across gaps.
    gap_bridge_cost : float
        Extra cost per pixel for gap bridging.
    tangent_window : int
        Number of recent steps to average for tangent continuity.
    distance_transform : np.ndarray or None
        Pre-computed EDT of binary image.  If given, the tracer prefers
        the center of strokes (higher DT values = lower cost), which
        prevents jumping to a neighboring parallel curve.

    Returns
    -------
    List[(x, y)] or None
        Ordered polyline from start to end, or None if no path found.
    """
    h, w = skeleton.shape

    # Snap anchors to nearest skeleton pixel
    start_snapped = _snap_to_skeleton(skeleton, start, snap_radius)
    end_snapped = _snap_to_skeleton(skeleton, end, snap_radius)

    if start_snapped is None or end_snapped is None:
        logger.warning("trace_with_anchors: could not snap anchor(s) to skeleton")
        return None

    sx, sy = start_snapped
    ex, ey = end_snapped

    logger.debug("trace_with_anchors: start=(%d,%d)->(%d,%d), end=(%d,%d)->(%d,%d)",
                 start[0], start[1], sx, sy, end[0], end[1], ex, ey)

    # Precompute max DT value for cost normalization
    dt_max = 1.0
    if distance_transform is not None:
        dt_max = max(float(distance_transform.max()), 1.0)

    def heuristic(x, y):
        return math.sqrt((x - ex) ** 2 + (y - ey) ** 2)

    # Priority queue: (f_score, g_score, x, y, prev_x, prev_y)
    open_set = [(heuristic(sx, sy), 0.0, sx, sy, -1, -1)]
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], float] = {(sx, sy): 0.0}
    closed: set = set()

    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    found = False
    max_iters = w * h * 2

    for _ in range(max_iters):
        if not open_set:
            break

        f, g, cx, cy, px, py = heapq.heappop(open_set)

        if (cx, cy) in closed:
            continue
        closed.add((cx, cy))
        came_from[(cx, cy)] = (px, py)

        if cx == ex and cy == ey:
            found = True
            break

        for dx, dy in dirs:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if (nx, ny) in closed:
                continue

            step_dist = math.sqrt(dx * dx + dy * dy)

            if skeleton[ny, nx]:
                # On skeleton — low cost
                move_cost = step_dist
                # Distance-transform bonus: prefer center of stroke
                if distance_transform is not None:
                    dt_val = float(distance_transform[ny, nx])
                    # Higher DT = more centered = lower cost
                    center_bonus = 0.3 * (1.0 - dt_val / dt_max)
                    move_cost += center_bonus
            else:
                # Off skeleton — gap bridging
                dist_to_skel = _distance_to_skeleton(skeleton, nx, ny, gap_bridge_max)
                if dist_to_skel > gap_bridge_max:
                    continue
                move_cost = step_dist * gap_bridge_cost

            # Curvature + tangent continuity penalty
            if px >= 0 and py >= 0:
                prev_dx = cx - px
                prev_dy = cy - py
                dot = prev_dx * dx + prev_dy * dy
                cross = abs(prev_dx * dy - prev_dy * dx)
                if dot < 0:
                    move_cost += curvature_penalty * 3.0  # strong U-turn penalty
                elif cross > 0:
                    move_cost += curvature_penalty * 0.3

            new_g = g + move_cost
            if new_g < g_score.get((nx, ny), float("inf")):
                g_score[(nx, ny)] = new_g
                f_score = new_g + heuristic(nx, ny)
                heapq.heappush(open_set, (f_score, new_g, nx, ny, cx, cy))

    if not found:
        logger.warning("trace_with_anchors: no path found between anchors")
        return None

    # Reconstruct path from A* back-pointers
    path = []
    cur = (ex, ey)
    while cur != (-1, -1) and cur in came_from:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()

    # ------------------------------------------------------------------
    # Anchor the polyline exactly at the user-selected start / end
    # coordinates (the original, un-snapped positions).  This guarantees
    # the returned polyline begins and ends EXACTLY where the user
    # clicked, regardless of how far the snap moved the point.
    # ------------------------------------------------------------------
    orig_start = (start[0], start[1])
    orig_end = (end[0], end[1])

    # Prepend original start if different from first path point
    if path and path[0] != orig_start:
        path.insert(0, orig_start)
    elif not path:
        path = [orig_start]

    # Append original end if different from last path point
    if path[-1] != orig_end:
        path.append(orig_end)

    logger.info("trace_with_anchors: found path with %d pixels (anchored)", len(path))
    return path


def _snap_to_skeleton(
    skeleton: np.ndarray,
    point: Tuple[int, int],
    radius: int,
) -> Optional[Tuple[int, int]]:
    """Snap a point to the nearest skeleton pixel within radius."""
    h, w = skeleton.shape
    px, py = point

    # Check exact point first
    if 0 <= py < h and 0 <= px < w and skeleton[py, px]:
        return (px, py)

    best = None
    best_dist = float("inf")

    for r in range(1, radius + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dy) != r and abs(dx) != r:
                    continue  # Only check border of square
                ny, nx = py + dy, px + dx
                if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < best_dist:
                        best_dist = dist
                        best = (nx, ny)
        if best is not None:
            return best

    return None


def _distance_to_skeleton(
    skeleton: np.ndarray,
    x: int, y: int,
    max_dist: int,
) -> int:
    """Find distance from (x,y) to nearest skeleton pixel, up to max_dist."""
    h, w = skeleton.shape
    for r in range(1, max_dist + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dy) != r and abs(dx) != r:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
                    return r
    return max_dist + 1


# ====================================================================
# 6. Curve endpoint extension
# ====================================================================

def extend_curve_ends(
    curve_pixels: List[Tuple[int, int]],
    binary: np.ndarray,
    *,
    search_radius: int = 20,
    cone_angle: float = 45.0,
    max_extension: int = 100,
    tangent_window: int = 10,
) -> List[Tuple[int, int]]:
    """Extend curve endpoints toward plot boundaries.

    For each endpoint:
      1. Estimate tangent direction from last N points
      2. Search forward in a cone for continuation pixels
      3. Follow the closest continuation
      4. Stop at plot boundary or when no continuation found

    Parameters
    ----------
    curve_pixels : list of (x, y)
        Ordered curve pixels.
    binary : np.ndarray (bool)
        Binary image (cleaned, before skeletonize).
    search_radius : int
        Forward search distance per step.
    cone_angle : float
        Half-angle of search cone (degrees).
    max_extension : int
        Maximum pixels to extend per end.
    tangent_window : int
        Number of endpoint pixels to estimate tangent from.

    Returns
    -------
    List[(x, y)]
        Extended curve pixels.
    """
    if len(curve_pixels) < 5:
        return curve_pixels

    h, w = binary.shape
    result = list(curve_pixels)

    # Extend from start (leftward)
    start_ext = _extend_one_end(
        result[:tangent_window], binary, h, w, reverse=True,
        search_radius=search_radius, cone_angle=cone_angle,
        max_extension=max_extension,
    )
    if start_ext:
        result = start_ext[::-1] + result

    # Extend from end (rightward)
    end_ext = _extend_one_end(
        result[-tangent_window:], binary, h, w, reverse=False,
        search_radius=search_radius, cone_angle=cone_angle,
        max_extension=max_extension,
    )
    if end_ext:
        result = result + end_ext

    return result


def _extend_one_end(
    endpoint_pixels: List[Tuple[int, int]],
    binary: np.ndarray,
    h: int, w: int,
    reverse: bool,
    *,
    search_radius: int,
    cone_angle: float,
    max_extension: int,
) -> List[Tuple[int, int]]:
    """Extend from one end of the curve."""
    if len(endpoint_pixels) < 3:
        return []

    # Get direction
    if reverse:
        pts = endpoint_pixels[::-1]
    else:
        pts = endpoint_pixels

    # Estimate tangent from last few points
    arr = np.array(pts[-min(10, len(pts)):], dtype=float)
    if len(arr) < 2:
        return []

    dx = arr[-1, 0] - arr[0, 0]
    dy = arr[-1, 1] - arr[0, 1]
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-6:
        return []

    dir_x = dx / length
    dir_y = dy / length

    # Current endpoint
    cx, cy = int(arr[-1, 0]), int(arr[-1, 1])

    # Check if already at boundary
    if cx <= 1 or cx >= w - 2 or cy <= 1 or cy >= h - 2:
        return []

    extension = []
    cos_limit = math.cos(math.radians(cone_angle))
    visited = set()

    for _ in range(max_extension):
        # Search in a forward cone for dark pixels
        best_px = None
        best_dist = float("inf")

        for sr in range(1, search_radius + 1):
            for angle_off in range(-int(cone_angle), int(cone_angle) + 1, 3):
                rad = math.radians(angle_off)
                sx = dir_x * math.cos(rad) - dir_y * math.sin(rad)
                sy = dir_x * math.sin(rad) + dir_y * math.cos(rad)

                nx = int(round(cx + sx * sr))
                ny = int(round(cy + sy * sr))

                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if (nx, ny) in visited:
                    continue
                if binary[ny, nx]:
                    dist = math.sqrt((nx - cx) ** 2 + (ny - cy) ** 2)
                    if dist < best_dist:
                        best_dist = dist
                        best_px = (nx, ny)

            if best_px is not None:
                break

        if best_px is None:
            break

        extension.append(best_px)
        visited.add(best_px)

        # Update direction
        old_cx, old_cy = cx, cy
        cx, cy = best_px
        new_dx = cx - old_cx
        new_dy = cy - old_cy
        new_len = math.sqrt(new_dx * new_dx + new_dy * new_dy)
        if new_len > 0.5:
            # Smooth direction update (80% old, 20% new)
            dir_x = 0.8 * dir_x + 0.2 * (new_dx / new_len)
            dir_y = 0.8 * dir_y + 0.2 * (new_dy / new_len)
            rlen = math.sqrt(dir_x ** 2 + dir_y ** 2)
            if rlen > 0:
                dir_x /= rlen
                dir_y /= rlen

        # Stop at boundary
        if cx <= 1 or cx >= w - 2 or cy <= 1 or cy >= h - 2:
            break

    return extension


# ====================================================================
# 7. Smoothing (Savitzky-Golay)
# ====================================================================

def smooth_curve(
    points: List[Tuple[float, float]],
    *,
    window_length: int = 0,
    polyorder: int = 3,
    resample_step: float = 0.0,
) -> List[Tuple[float, float]]:
    """Apply Savitzky-Golay smoothing to a polyline.

    Parameters
    ----------
    points : list of (x, y)
        Ordered polyline coordinates (axis or pixel space).
    window_length : int
        SG filter window (must be odd, > polyorder).
        0 = auto-select based on point count.
    polyorder : int
        Polynomial order for SG filter.
    resample_step : float
        If > 0, resample to uniform x-step before smoothing.

    Returns
    -------
    List[(x, y)]
        Smoothed polyline.
    """
    if len(points) < 5:
        return points

    arr = np.array(sorted(points, key=lambda p: p[0]), dtype=float)
    xs, ys = arr[:, 0], arr[:, 1]

    # Deduplicate x
    ux, idx = np.unique(xs, return_index=True)
    ys = ys[idx]
    xs = ux

    if len(xs) < 5:
        return [(float(x), float(y)) for x, y in zip(xs, ys)]

    # Optional resampling
    if resample_step > 0:
        new_xs = np.arange(xs[0], xs[-1], resample_step)
        if len(new_xs) < 5:
            new_xs = np.linspace(xs[0], xs[-1], max(5, len(xs)))
        ys = np.interp(new_xs, xs, ys)
        xs = new_xs

    # Auto window length
    if window_length <= 0:
        window_length = max(5, min(len(xs) // 5, 51))
    window_length = min(window_length, len(xs))
    if window_length % 2 == 0:
        window_length += 1
    if window_length <= polyorder:
        window_length = polyorder + 2
        if window_length % 2 == 0:
            window_length += 1

    try:
        from scipy.signal import savgol_filter
        ys_smooth = savgol_filter(ys, window_length, polyorder)
    except Exception:
        # Fallback: simple moving average
        kernel = np.ones(min(5, len(ys))) / min(5, len(ys))
        ys_smooth = np.convolve(ys, kernel, mode="same")

    return [(float(x), float(y)) for x, y in zip(xs, ys_smooth)]


# ====================================================================
# 8. Order pixels into polyline (for skeleton components)
# ====================================================================

def order_pixels_to_polyline(
    pixels: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Order skeleton pixels into a contiguous polyline.

    Uses a greedy nearest-neighbor walk starting from the leftmost pixel.
    """
    if len(pixels) < 3:
        return sorted(pixels, key=lambda p: p[0])

    # Build a set for O(1) lookup
    pixel_set = set(pixels)
    remaining = set(pixels)

    # Find leftmost pixel as start
    start = min(pixels, key=lambda p: (p[0], p[1]))
    path = [start]
    remaining.discard(start)

    while remaining:
        cx, cy = path[-1]
        best = None
        best_dist = float("inf")

        # Check 8-neighbors first
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in remaining:
                    dist = abs(dx) + abs(dy)
                    if dist < best_dist:
                        best_dist = dist
                        best = (nx, ny)

        if best is None:
            # No immediate neighbor — find closest remaining
            for px, py in remaining:
                dist = abs(px - cx) + abs(py - cy)
                if dist < best_dist:
                    best_dist = dist
                    best = (px, py)
            if best is None or best_dist > 30:
                break

        path.append(best)
        remaining.discard(best)

    return path


# ====================================================================
# 8b. Column-scan multi-curve extraction
# ====================================================================

def _column_scan_extract(
    binary: np.ndarray,
    num_curves: int = 5,
    *,
    max_y_jump: int = 0,
    min_track_width: int = 0,
    max_x_gap: int = 0,
    min_coverage: float = 0.40,
    max_slope: float = 3.0,
) -> Dict[int, List[Tuple[int, int]]]:
    """Extract curves from a binary image using column-scan tracking.

    Scans each column left-to-right, finds vertical runs of dark pixels,
    and tracks their centroids across columns.  Uses slope prediction to
    correctly separate overlapping curves.

    This method naturally handles:
      - Multiple overlapping solid curves (different y positions)
      - Curve crossings (tracks maintain identity through crossings
        via slope prediction)
      - Small gaps (tracks coast through using predicted slope)

    Parameters
    ----------
    binary : np.ndarray (bool)
        Cleaned binary mask (text, grid already removed).
    num_curves : int
        Expected number of curves (upper bound for output).
    max_y_jump : int
        Max y deviation from predicted position.  0 = auto.
    min_track_width : int
        Min horizontal span for a valid track.  0 = auto.
    max_x_gap : int
        Max columns without a match before a track goes stale.  0 = auto.
    min_coverage : float
        Min fraction of x-span that must have data.
    max_slope : float
        Max y_span / x_span ratio (reject near-vertical tracks).

    Returns
    -------
    Dict[int, List[(x, y)]]
        Curve index → ordered (x, y) pixel coordinates (region-local).
    """
    rh, rw = binary.shape

    # Auto-compute parameters from image dimensions
    if max_y_jump <= 0:
        max_y_jump = max(8, int(rh * 0.03))
    if min_track_width <= 0:
        min_track_width = max(10, int(rw * 0.15))
    if max_x_gap <= 0:
        max_x_gap = max(20, int(rw * 0.10))

    # --- Step 1: Find dark-pixel runs in every column ---
    RUN_GAP = max(3, int(rh * 0.005))
    column_runs: Dict[int, list] = {}

    for x in range(rw):
        dark_y = np.where(binary[:, x])[0]
        if len(dark_y) == 0:
            continue
        runs = []
        start = dark_y[0]
        for i in range(1, len(dark_y)):
            if dark_y[i] - dark_y[i - 1] > RUN_GAP:
                runs.append((start, dark_y[i - 1]))
                start = dark_y[i]
        runs.append((start, dark_y[-1]))
        column_runs[x] = [
            (int((r[0] + r[1]) // 2), int(r[0]), int(r[1])) for r in runs
        ]

    if not column_runs:
        return {}

    # --- Step 2: Slope-predicting nearest-neighbor tracker ---
    SLOPE_WINDOW = 5
    tracks: list = []
    track_last_x: list = []
    track_last_y: list = []
    track_slope: list = []

    def _estimate_slope(track_pts):
        if len(track_pts) < 2:
            return 0.0
        recent = track_pts[-SLOPE_WINDOW:]
        if len(recent) < 2:
            return 0.0
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        return dy / dx if dx != 0 else 0.0

    for x in sorted(column_runs.keys()):
        centroids = [r[0] for r in column_runs[x]]
        if not tracks:
            for cy in centroids:
                tracks.append([(x, cy)])
                track_last_x.append(x)
                track_last_y.append(cy)
                track_slope.append(0.0)
            continue

        # Match centroids to existing tracks
        used_tracks: set = set()
        used_cents: set = set()
        pairs = []
        for ci, cy in enumerate(centroids):
            for ti in range(len(tracks)):
                x_gap = x - track_last_x[ti]
                if x_gap > max_x_gap:
                    continue
                predicted_y = track_last_y[ti] + track_slope[ti] * x_gap
                dist = abs(cy - predicted_y)
                if dist <= max_y_jump:
                    pairs.append((dist, ci, ti))
        pairs.sort()

        for dist, ci, ti in pairs:
            if ci in used_cents or ti in used_tracks:
                continue
            tracks[ti].append((x, centroids[ci]))
            track_last_x[ti] = x
            track_last_y[ti] = centroids[ci]
            track_slope[ti] = _estimate_slope(tracks[ti])
            used_cents.add(ci)
            used_tracks.add(ti)

        # Unmatched centroids → new tracks
        for ci, cy in enumerate(centroids):
            if ci not in used_cents:
                tracks.append([(x, cy)])
                track_last_x.append(x)
                track_last_y.append(cy)
                track_slope.append(0.0)

    # --- Step 3: Running-median smoothing ---
    SMOOTH_HALF = 5
    SMOOTH_THR = max(6, int(rh * 0.015))
    smoothed_tracks: list = []
    for t in tracks:
        if len(t) < SMOOTH_HALF * 2 + 1:
            smoothed_tracks.append(t)
            continue
        arr = np.array(t)
        ys = arr[:, 1].astype(float)
        keep = np.ones(len(ys), dtype=bool)
        for i in range(len(ys)):
            lo = max(0, i - SMOOTH_HALF)
            hi = min(len(ys), i + SMOOTH_HALF + 1)
            med = np.median(ys[lo:hi])
            if abs(ys[i] - med) > SMOOTH_THR:
                keep[i] = False
        cleaned = arr[keep]
        if len(cleaned) >= 5:
            smoothed_tracks.append([(int(p[0]), int(p[1])) for p in cleaned])
        else:
            smoothed_tracks.append(t)
    tracks = smoothed_tracks

    # --- Step 4: Filter tracks ---
    valid_tracks = []
    for t in tracks:
        xs_t = [p[0] for p in t]
        ys_t = [p[1] for p in t]
        x_span = max(xs_t) - min(xs_t) if xs_t else 0
        y_span = (max(ys_t) - min(ys_t)) if ys_t else 0

        if x_span < min_track_width:
            continue

        slope = y_span / max(x_span, 1)
        if slope > max_slope:
            continue

        unique_x = len(set(xs_t))
        coverage = unique_x / max(x_span, 1)
        if coverage < min_coverage:
            continue

        valid_tracks.append(t)

    # --- Step 5: Rank and limit ---
    def _track_score(t):
        xs_t = [p[0] for p in t]
        x_span = max(xs_t) - min(xs_t)
        unique_x = len(set(xs_t))
        return x_span * (unique_x / max(x_span, 1))

    valid_tracks.sort(key=_track_score, reverse=True)
    if len(valid_tracks) > num_curves:
        valid_tracks = valid_tracks[:num_curves]

    # Sort by mean y (topmost first)
    valid_tracks.sort(key=lambda t: float(np.mean([p[1] for p in t])))

    result: Dict[int, List[Tuple[int, int]]] = {}
    for idx, track in enumerate(valid_tracks):
        result[idx] = track

    logger.debug("_column_scan_extract: found %d tracks", len(result))
    return result


# ====================================================================
# 8c. Curve exclusion filter
# ====================================================================

def _exclude_curve_filter(
    curves: Dict[int, List[Tuple[int, int]]],
    mode: str,
) -> Dict[int, List[Tuple[int, int]]]:
    """Exclude one curve based on a configurable heuristic.

    Use this to remove a "surge line" or other reference line from
    the extracted curves without hardcoding image-specific rules.

    Modes
    -----
    topmost     – exclude the curve with lowest mean y (highest on screen)
    bottommost  – exclude the curve with highest mean y
    steepest    – exclude the curve with largest y_span / x_span ratio
    longest     – exclude the curve with largest x_span
    thickest    – exclude the curve with most pixels

    Parameters
    ----------
    curves : Dict[int, List[(x, y)]]
    mode : str
        One of the modes above, or '' (no-op).

    Returns
    -------
    Dict[int, List[(x, y)]]
        Re-indexed dict with the excluded curve removed.
    """
    if not mode or len(curves) <= 1:
        return curves

    mode = mode.lower().strip()
    valid_modes = ("topmost", "bottommost", "steepest", "longest", "thickest")
    if mode not in valid_modes:
        logger.warning("_exclude_curve_filter: unknown mode '%s'", mode)
        return curves

    scores: Dict[int, float] = {}
    for idx, pixels in curves.items():
        if not pixels:
            continue
        xs = [p[0] for p in pixels]
        ys = [p[1] for p in pixels]
        mean_y = float(np.mean(ys))
        x_span = max(xs) - min(xs)
        y_span = max(ys) - min(ys)
        slope = y_span / max(x_span, 1)

        if mode == "topmost":
            scores[idx] = -mean_y
        elif mode == "bottommost":
            scores[idx] = mean_y
        elif mode == "steepest":
            scores[idx] = slope
        elif mode == "longest":
            scores[idx] = float(x_span)
        elif mode == "thickest":
            scores[idx] = float(len(pixels))

    if not scores:
        return curves

    exclude_idx = max(scores, key=scores.get)  # type: ignore[arg-type]
    logger.info(
        "_exclude_curve_filter: excluding curve %d (mode=%s, score=%.2f)",
        exclude_idx, mode, scores[exclude_idx],
    )

    # Rebuild without the excluded curve, re-indexing from 0
    remaining = {k: v for k, v in curves.items() if k != exclude_idx}
    result: Dict[int, List[Tuple[int, int]]] = {}
    for new_idx, (_, pixels) in enumerate(sorted(remaining.items())):
        result[new_idx] = pixels

    return result


# ====================================================================
# 9. Full B/W extraction pipeline
# ====================================================================

def extract_bw_curves(
    image: Image.Image,
    num_curves: int,
    plot_area: Tuple[int, int, int, int],
    *,
    anchors: Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]] = None,
    ignore_dashed: bool = True,
    dashed_threshold: float = 0.45,
    text_threshold: float = 0.50,
    smoothing_strength: int = 0,
    extend_ends: bool = True,
    exclude_curve_mode: str = "",
    config: Optional[BWPipelineConfig] = None,
) -> Dict[int, List[Tuple[int, int]]]:
    """Full B/W curve extraction pipeline.

    Pipeline:
      1. Preprocess (binarize, remove grid/text/ticks, skeletonize)
      2. If anchors: trace between anchors using A* with distance-transform
         Else: column-scan multi-curve extraction on cleaned binary
         Fallback: iterative skeleton component selection
      3. Extend curve endpoints toward boundaries
      4. Smooth the output
      5. Optionally exclude one curve (surge line filter)

    Parameters
    ----------
    image : PIL.Image
    num_curves : int
        Expected number of curves.
    plot_area : (left, top, right, bottom)
    anchors : list of ((start_x, start_y), (end_x, end_y)) per curve, or None
        Pixel coordinates in full-image space.
    ignore_dashed : bool
        Down-rank (soft) dashed/dotted lines; do NOT delete them.
    dashed_threshold : float
        Threshold for dashed-score soft penalty.
    text_threshold : float
        Threshold for text-score soft penalty.
    smoothing_strength : int
        Savitzky-Golay window (0 = auto, -1 = disabled).
    extend_ends : bool
        Whether to extend curve endpoints.
    exclude_curve_mode : str
        If non-empty, exclude one curve by mode:
        'topmost', 'bottommost', 'steepest', 'longest', 'thickest'.
        Use 'steepest' to remove a surge line.

    Returns
    -------
    Dict[int, List[(x, y)]]
        Curve index → pixel coords in full-image space.
    """
    cfg = config or DEFAULT_CONFIG
    p_left, p_top, p_right, p_bottom = plot_area

    # Step 1: Preprocess (enhanced with CLAHE, blackhat, adaptive, Hough)
    skeleton, binary, adj_area = preprocess_bw(image, plot_area, config=cfg)
    rh, rw = skeleton.shape

    # Pre-compute distance transform on binary for anchor tracing
    dt = distance_transform_edt(binary)

    # Step 2: Extract curves (hybrid)
    result: Dict[int, List[Tuple[int, int]]] = {}
    target = max(num_curves, 1)
    working_skeleton = skeleton.copy()

    def _mask_pixels(mask_img: np.ndarray, pixels_local: List[Tuple[int, int]], radius: int = 2) -> None:
        for px, py in pixels_local:
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < rh and 0 <= nx < rw:
                        mask_img[ny, nx] = False

    # ── Step 2-pre: DP-based ordered multi-curve extraction ──
    #    Extracts non-crossing curves via sequential DP with exclusion bands.
    #    Skip when anchors are provided – anchor-guided tracing is more precise.
    if not anchors:
        try:
            from core.dp_tracker import extract_curves_dp

            dp_curves, dp_debug = extract_curves_dp(
                binary, skeleton, target,
                max_jump=0,          # auto
                jump_weight=0.3,
                curvature_weight=0.1,
                min_span_ratio=0.25,
            )
            if dp_curves:
                for idx, pixels in dp_curves.items():
                    result[idx] = [(x + p_left, y + p_top)
                                   for x, y in pixels]
                    _mask_pixels(working_skeleton, pixels, radius=2)
                logger.info(
                    "extract_bw_curves: DP tracker produced %d curves",
                    len(dp_curves),
                )
                # Save debug overlay when requested
                if cfg.debug_bw and cfg.debug_bw_dir:
                    try:
                        from core.reconstruction import render_dp_debug
                        render_dp_debug(
                            np.array(image), plot_area, skeleton,
                            dp_curves, dp_debug, cfg.debug_bw_dir,
                        )
                    except Exception as dbg_err:
                        logger.debug("DP debug overlay failed: %s", dbg_err)
        except Exception as dp_err:
            logger.warning("DP tracker failed, falling back: %s", dp_err)

    # 2a) Anchor-guided tracing (if provided)
    if anchors:
        for start, end in anchors:
            if len(result) >= target:
                break

            local_start = (start[0] - p_left, start[1] - p_top)
            local_end = (end[0] - p_left, end[1] - p_top)

            path = trace_with_anchors(
                working_skeleton, local_start, local_end,
                distance_transform=dt,
            )
            if not path:
                continue

            idx = len(result)
            full_path = [(x + p_left, y + p_top) for x, y in path]
            result[idx] = full_path

            # Remove traced path from working skeleton
            _mask_pixels(working_skeleton, path, radius=2)

    # 2b) Iterative skeleton component selection (proven primary method)
    remaining = target - len(result)
    if remaining > 0:
        max_rounds = remaining + 3
        for _round in range(max_rounds):
            if len(result) >= target:
                break

            components = extract_skeleton_components(working_skeleton)
            if not components:
                break

            selected = select_best_curves(
                components, 1,
                dashed_threshold=dashed_threshold,
                text_threshold=text_threshold,
                ignore_dashed=ignore_dashed,
                auto_detect=False,
            )
            if not selected:
                break

            comp = selected[0]
            idx = len(result)
            raw_pixels = comp["pixels"]
            ordered = order_pixels_to_polyline(raw_pixels)
            full_pixels = [(x + p_left, y + p_top) for x, y in ordered]
            result[idx] = full_pixels

            _mask_pixels(working_skeleton, raw_pixels, radius=2)

            logger.debug(
                "Skeleton extraction round %d: component %d px, x_span=%d",
                _round, comp["area"], comp["x_span"],
            )

    # 2c) Column-scan fallback: if skeleton components didn't find enough
    #     curves (e.g., overlapping curves in one connected component),
    #     use the column-scan tracker on the cleaned binary.
    remaining = target - len(result)
    if remaining > 0:
        work_binary = binary.copy()
        for idx_exist in result:
            local_pxs = [(x - p_left, y - p_top) for x, y in result[idx_exist]]
            _mask_pixels(work_binary, local_pxs, radius=3)

        scanned = _column_scan_extract(work_binary, remaining)
        for scan_idx, pixels in sorted(scanned.items()):
            if len(result) >= target:
                break
            idx = len(result)
            full_pixels = [(x + p_left, y + p_top) for x, y in pixels]
            result[idx] = full_pixels

        if scanned:
            logger.debug(
                "Column-scan fallback found %d additional curve(s)",
                len(scanned),
            )

    # 2d) Geometry sanity filter: remove likely non-curve tracks
    #      (e.g., near-vertical reference/flow lines) and keep best tracks.
    if result:
        def _track_stats(pixels: List[Tuple[int, int]]) -> Dict[str, float]:
            xs = np.array([p[0] for p in pixels], dtype=np.float64)
            ys = np.array([p[1] for p in pixels], dtype=np.float64)
            if len(xs) == 0:
                return {
                    "x_span": 0.0,
                    "y_span": 0.0,
                    "coverage": 0.0,
                    "slope_ratio": 0.0,
                    "score": 0.0,
                }
            x_span = float(xs.max() - xs.min())
            y_span = float(ys.max() - ys.min())
            coverage = float(len(np.unique(xs.astype(int))) / max(x_span, 1.0))
            slope_ratio = float(y_span / max(x_span, 1.0))
            score = x_span * coverage
            return {
                "x_span": x_span,
                "y_span": y_span,
                "coverage": coverage,
                "slope_ratio": slope_ratio,
                "score": score,
            }

        # Hard reject obvious non-curves
        all_items: List[Tuple[int, List[Tuple[int, int]], Dict[str, float]]] = []
        filtered_items: List[Tuple[int, List[Tuple[int, int]], Dict[str, float]]] = []
        rejected_items: List[Tuple[int, List[Tuple[int, int]], Dict[str, float]]] = []
        for idx, pixels in result.items():
            st = _track_stats(pixels)
            all_items.append((idx, pixels, st))
            if st["x_span"] < rw * 0.20:
                rejected_items.append((idx, pixels, st))
                continue
            if st["coverage"] < 0.18:
                rejected_items.append((idx, pixels, st))
                continue
            # Very steep + not wide enough -> likely reference/flow line
            if st["slope_ratio"] > 1.2 and st["x_span"] < rw * 0.70:
                rejected_items.append((idx, pixels, st))
                continue
            filtered_items.append((idx, pixels, st))

        if filtered_items:
            # If too few survive, backfill with best rejected tracks.
            if len(filtered_items) < target and rejected_items:
                rejected_items.sort(key=lambda t: t[2]["score"], reverse=True)
                need = target - len(filtered_items)
                filtered_items.extend(rejected_items[:need])

            # If still over target, keep the strongest spanning tracks
            filtered_items.sort(key=lambda t: t[2]["score"], reverse=True)
            filtered_items = filtered_items[:target]
            result = {i: px for i, (_, px, _) in enumerate(filtered_items)}
        else:
            # If all were filtered out, keep original as safety fallback
            logger.debug("Geometry sanity filter dropped all tracks; keeping original set")

    # Step 3: Extend curves toward boundaries (skip if anchors given —
    #          the user's anchors define exact bounds)
    if extend_ends and not anchors:
        for idx in list(result.keys()):
            pixels = result[idx]
            local_pixels = [(x - p_left, y - p_top) for x, y in pixels]
            extended = extend_curve_ends(local_pixels, binary)
            result[idx] = [(x + p_left, y + p_top) for x, y in extended]

    # Step 4: Smooth
    if smoothing_strength >= 0:
        for idx in list(result.keys()):
            pixels = result[idx]
            if len(pixels) < 5:
                continue
            # Preserve first and last points (anchored endpoints)
            first_pt = pixels[0]
            last_pt = pixels[-1]
            smoothed = smooth_curve(
                [(float(x), float(y)) for x, y in pixels],
                window_length=smoothing_strength,
            )
            smoothed_int = [(int(round(x)), int(round(y))) for x, y in smoothed]
            # Re-anchor first/last to exact positions
            if smoothed_int:
                smoothed_int[0] = first_pt
                smoothed_int[-1] = last_pt
            result[idx] = smoothed_int

    # Step 5: Exclude one curve if requested (e.g., surge line)
    if exclude_curve_mode:
        result = _exclude_curve_filter(result, exclude_curve_mode)

    # Step 6: Re-index by mean y (topmost first) for consistent ordering
    if len(result) > 1:
        sorted_items = sorted(
            result.items(),
            key=lambda kv: float(np.mean([p[1] for p in kv[1]])) if kv[1] else 0,
        )
        result = {i: pixels for i, (_, pixels) in enumerate(sorted_items)}

    # ── Debug visualisation ──
    _save_bw_debug_overlay(
        image, plot_area, skeleton, binary, result, anchors,
    )

    logger.info("extract_bw_curves: extracted %d curves", len(result))
    return result


def _save_bw_debug_overlay(
    image: Image.Image,
    plot_area: Tuple[int, int, int, int],
    skeleton: np.ndarray,
    binary: np.ndarray,
    curves: Dict[int, List[Tuple[int, int]]],
    anchors: Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]],
) -> None:
    """Save a composite debug image with overlays for diagnostics.

    Saved artefacts (all under ``_DEBUG_DIR``):
      - ``bw_debug_composite.png``  : plot-rect, skeleton, curves, anchors
      - ``bw_debug_binary.png``     : cleaned binary mask
      - ``bw_debug_skeleton.png``   : skeleton
    """
    if not _DEBUG_DIR:
        return
    try:
        import copy
        from PIL import ImageDraw

        p_left, p_top, p_right, p_bottom = plot_area

        # --- Composite overlay on original image ---
        overlay = image.convert("RGBA").copy()
        draw = ImageDraw.Draw(overlay)

        # Plot rectangle (cyan)
        draw.rectangle(
            [p_left, p_top, p_right - 1, p_bottom - 1],
            outline=(0, 255, 255, 200),
            width=2,
        )

        # Curve polylines
        curve_colors = [
            (255, 0, 0, 230),    # red
            (0, 200, 0, 230),    # green
            (0, 80, 255, 230),   # blue
            (255, 165, 0, 230),  # orange
            (180, 0, 255, 230),  # purple
        ]
        for idx, pixels in curves.items():
            c = curve_colors[idx % len(curve_colors)]
            for i in range(len(pixels) - 1):
                draw.line([pixels[i], pixels[i + 1]], fill=c, width=2)

        # Anchor markers (start = green circle, end = red circle)
        if anchors:
            for start, end in anchors:
                r = 6
                draw.ellipse(
                    [start[0] - r, start[1] - r, start[0] + r, start[1] + r],
                    outline=(0, 255, 0, 255), width=2,
                )
                draw.ellipse(
                    [end[0] - r, end[1] - r, end[0] + r, end[1] + r],
                    outline=(255, 0, 0, 255), width=2,
                )

        out = Path(_DEBUG_DIR)
        out.mkdir(parents=True, exist_ok=True)
        overlay.save(str(out / "bw_debug_composite.png"))

        # --- Skeleton overlay ---
        skel_img = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
        skel_img[skeleton] = [0, 255, 0]
        for idx, pixels in curves.items():
            c_rgb = curve_colors[idx % len(curve_colors)][:3]
            for px, py in pixels:
                lx, ly = px - p_left, py - p_top
                if 0 <= ly < skel_img.shape[0] and 0 <= lx < skel_img.shape[1]:
                    skel_img[ly, lx] = c_rgb
        Image.fromarray(skel_img).save(str(out / "bw_debug_skeleton_curves.png"))

    except Exception as exc:
        logger.debug("_save_bw_debug_overlay failed: %s", exc)
