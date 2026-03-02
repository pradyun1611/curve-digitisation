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
    binary_opening,
    distance_transform_edt,
    label as ndimage_label,
)

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
    text_area_max_ratio: float = 0.005,
    text_aspect_min: float = 0.3,
    text_aspect_max: float = 3.5,
    close_kernel_size: Tuple[int, int] = (3, 7),
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
    img_array = np.array(image.convert("RGB"))
    p_left, p_top, p_right, p_bottom = plot_area

    # Crop to plot area
    region = img_array[p_top:p_bottom, p_left:p_right]
    if region.ndim == 3:
        gray = np.mean(region[:, :, :3].astype(np.float32), axis=2).astype(np.uint8)
    else:
        gray = region.astype(np.uint8)
    rh, rw = gray.shape

    # Denoise
    gray = np.array(Image.fromarray(gray).filter(ImageFilter.MedianFilter(size=3)))

    # Adaptive Otsu threshold
    threshold = _otsu_threshold(gray)
    binary = gray <= min(threshold + 10, 200)
    _save_debug("preprocess_binary", binary.astype(np.uint8) * 255)

    # Morphological close to bridge small gaps
    ck = np.ones(close_kernel_size, dtype=bool)
    binary = binary_closing(binary, structure=ck, iterations=1)

    # Remove grid lines (rows/cols with >75% fill)
    row_fill = binary.sum(axis=1) / rw
    binary[row_fill > 0.75, :] = False
    col_fill = binary.sum(axis=0) / rh
    binary[:, col_fill > 0.75] = False

    # Remove isolated single-pixel noise (but preserve thin curves)
    # Instead of morphological opening, remove pixels with 0 or 1 neighbors
    ys, xs = np.where(binary)
    if len(ys) > 0:
        padded = np.pad(binary.astype(np.uint8), 1, mode='constant')
        neighbor_count = np.zeros_like(binary, dtype=np.uint8)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighbor_count += padded[1 + dy:padded.shape[0] - 1 + dy,
                                         1 + dx:padded.shape[1] - 1 + dx]
        # Remove truly isolated pixels (0 neighbors)
        binary[neighbor_count == 0] = False

    # Remove text via connected-component analysis
    binary = _remove_text_components(binary, rh, rw,
                                     area_max_ratio=text_area_max_ratio,
                                     aspect_min=text_aspect_min,
                                     aspect_max=text_aspect_max)
    _save_debug("preprocess_no_text", binary.astype(np.uint8) * 255)

    # Remove axis remnants at borders
    binary = _remove_border_lines(binary, border_px=5)

    # Skeletonize
    skeleton = _skeletonize(binary)
    _save_debug("preprocess_skeleton", skeleton.astype(np.uint8) * 255)

    return skeleton, binary, (p_left, p_top, p_right, p_bottom)


def _remove_text_components(
    binary: np.ndarray,
    img_h: int,
    img_w: int,
    *,
    area_max_ratio: float = 0.005,
    aspect_min: float = 0.3,
    aspect_max: float = 3.5,
    min_skeleton_length: int = 30,
) -> np.ndarray:
    """Remove connected components that are likely text labels.

    Text heuristics:
      - Small area relative to plot region
      - Bounding-box aspect ratio typical of characters (0.3-3.5)
      - Not part of a long continuous structure
      - Located near margins
    """
    structure = np.ones((3, 3), dtype=int)
    labelled, n_comp = ndimage_label(binary, structure=structure)

    if n_comp == 0:
        return binary

    plot_area_px = img_h * img_w
    area_max = int(plot_area_px * area_max_ratio)

    result = binary.copy()
    for comp_id in range(1, n_comp + 1):
        comp_mask = labelled == comp_id
        area = int(comp_mask.sum())

        if area > area_max:
            # Too large to be text — keep it
            continue

        if area < 3:
            # Tiny speck — remove
            result[comp_mask] = False
            continue

        # Bounding box
        ys, xs = np.where(comp_mask)
        bbox_h = int(ys.max() - ys.min()) + 1
        bbox_w = int(xs.max() - xs.min()) + 1
        aspect = bbox_w / max(bbox_h, 1)

        # Check if it's a character-like shape
        if aspect_min <= aspect <= aspect_max and area < area_max:
            # Check if near margins (outer 10% of plot area)
            margin_x = img_w * 0.10
            margin_y = img_h * 0.10
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            near_margin = (cx < margin_x or cx > img_w - margin_x or
                          cy < margin_y or cy > img_h - margin_y)

            # Density: text chars are usually dense
            density = area / (bbox_h * bbox_w)

            if near_margin or (density > 0.2 and bbox_w < img_w * 0.08):
                result[comp_mask] = False
                continue

        # Small isolated component not spanning much width
        x_span = bbox_w
        if x_span < min_skeleton_length and area < area_max * 0.5:
            result[comp_mask] = False

    return result


def _remove_border_lines(binary: np.ndarray, border_px: int = 5) -> np.ndarray:
    """Remove dark pixels within border_px of the image edges."""
    result = binary.copy()
    result[:border_px, :] = False
    result[-border_px:, :] = False
    result[:, :border_px] = False
    result[:, -border_px:] = False
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

    # Reconstruct path
    path = []
    cur = (ex, ey)
    while cur != (-1, -1) and cur in came_from:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()

    logger.info("trace_with_anchors: found path with %d pixels", len(path))
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
) -> Dict[int, List[Tuple[int, int]]]:
    """Full B/W curve extraction pipeline.

    Pipeline:
      1. Preprocess (binarize, remove text, skeletonize)
      2. SOFT score all components (no destructive deletion)
      3. If anchors: trace between anchors using A* with distance-transform
         Else: iterative multi-curve selection (select, mask, repeat)
      4. Extend curve endpoints toward boundaries
      5. Smooth the output

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
        Savitzky-Golay window (0 = auto).
    extend_ends : bool
        Whether to extend curve endpoints.

    Returns
    -------
    Dict[int, List[(x, y)]]
        Curve index → pixel coords in full-image space.
    """
    p_left, p_top, p_right, p_bottom = plot_area

    # Step 1: Preprocess
    skeleton, binary, adj_area = preprocess_bw(image, plot_area)
    rh, rw = skeleton.shape

    # NOTE: We do NOT call filter_dashed_components() here.
    # Dashed/text scoring is SOFT inside select_best_curves —
    # components are down-ranked, not destroyed.
    # This preserves real curves that may have been lost before.

    # Pre-compute distance transform on binary for anchor tracing
    dt = distance_transform_edt(binary)

    # Step 2: Extract or trace curves
    result: Dict[int, List[Tuple[int, int]]] = {}

    if anchors:
        # Anchor-guided tracing (primary for B/W)
        for idx, (start, end) in enumerate(anchors):
            local_start = (start[0] - p_left, start[1] - p_top)
            local_end = (end[0] - p_left, end[1] - p_top)

            path = trace_with_anchors(
                skeleton, local_start, local_end,
                distance_transform=dt,
            )
            if path:
                full_path = [(x + p_left, y + p_top) for x, y in path]
                result[idx] = full_path
    else:
        # Iterative multi-curve extraction:
        # 1. Extract components, pick the best one
        # 2. Mask its pixels from skeleton
        # 3. Repeat until we have enough curves or no good candidates remain
        working_skeleton = skeleton.copy()
        target = max(num_curves, 1)
        max_rounds = target + 3  # allow a few extra rounds for safety

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

            # Mask this curve's pixels (+ small dilation) from working skeleton
            for px, py in raw_pixels:
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ny, nx = py + dy, px + dx
                        if 0 <= ny < rh and 0 <= nx < rw:
                            working_skeleton[ny, nx] = False

            logger.debug(
                "Iterative extraction round %d: picked component with %d px, x_span=%d",
                _round, comp["area"], comp["x_span"],
            )

    # Step 3: Extend curves toward boundaries
    if extend_ends:
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
            smoothed = smooth_curve(
                [(float(x), float(y)) for x, y in pixels],
                window_length=smoothing_strength,
            )
            result[idx] = [(int(round(x)), int(round(y))) for x, y in smoothed]

    logger.info("extract_bw_curves: extracted %d curves", len(result))
    return result
