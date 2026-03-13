"""
Image Processing Module

Handles image digitization and curve fitting:
- Image preprocessing and cropping
- Curve color detection and extraction
- Pixel-to-coordinate normalization
- RANSAC noise removal
- Polynomial curve fitting
- Digitized graph generation
"""

import logging
import os
import numpy as np
from PIL import Image, ImageFilter
import json
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/headless use
import matplotlib.pyplot as plt
from datetime import datetime

from core.router import classify_image_mode
from core.bw_pipeline import extract_bw_curves, extract_surge_lines, smooth_curve, get_last_skeleton
from core.bw_fit import fit_bw_curve

# ── Determinism: single-threaded BLAS/LAPACK ──────────────────────
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
np.random.seed(42)

logger = logging.getLogger(__name__)

# ── Optional debug-image persistence ──────────────────────────────
_DEBUG_DIR = os.environ.get("CURVE_DEBUG_IMAGES", "")


def _save_debug_image(tag: str, array: np.ndarray) -> None:
    """Save a debug image to *_DEBUG_DIR* if the env-var is set."""
    if not _DEBUG_DIR:
        return
    try:
        out = Path(_DEBUG_DIR)
        out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).save(str(out / f"{tag}.png"))
    except Exception:
        pass


class CurveDigitizer:
    """Digitizes performance curves from images."""
    
    def __init__(self, axis_info: Dict[str, Any]):
        """
        Initialize digitizer with axis information.
        
        Args:
            axis_info: Dictionary with xMin, xMax, yMin, yMax, xUnit, yUnit
        """
        self.axis_info = axis_info
        self.xMin = axis_info.get('xMin', 0)
        self.xMax = axis_info.get('xMax', 100)
        self.yMin = axis_info.get('yMin', 0)
        self.yMax = axis_info.get('yMax', 100)
        self.xUnit = axis_info.get('xUnit', 'units')
        self.yUnit = axis_info.get('yUnit', 'units')
    
    def load_image(self, image_path: str) -> Image.Image:
        """
        Load image from file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image object
        """
        return Image.open(image_path).convert('RGB')
    
    def crop_image(self, image: Image.Image, crop_box: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
        """
        Crop/snip image to region of interest.
        
        Args:
            image: PIL Image object
            crop_box: Optional tuple (left, top, right, bottom) for cropping
            
        Returns:
            Cropped image
        """
        if crop_box is None:
            return image
        
        return image.crop(crop_box)
    
    def extract_color_pixels(self, image: Image.Image, target_color_name: str) -> List[Tuple[int, int]]:
        """
        Extract pixel coordinates for a specific color in the image.
        
        Args:
            image: PIL Image object
            target_color_name: Color name (e.g., 'red', 'blue', 'green')
            
        Returns:
            List of (x, y) pixel coordinates matching the color
        """
        # Convert to RGB numpy array
        img_array = np.array(image)
        
        # ── Try HSV-based extraction first (robust to JPEG/anti-aliasing) ──
        hsv_pixels = self._extract_via_hsv(img_array, target_color_name)
        if hsv_pixels is not None and len(hsv_pixels) >= 5:
            logger.debug("extract_color_pixels: HSV path returned %d px for '%s'",
                         len(hsv_pixels), target_color_name)
            _save_debug_image(f"color_hsv_{target_color_name}",
                              self._pixels_to_debug_mask(hsv_pixels, img_array.shape))
            return hsv_pixels

        # ── Fallback: RGB thresholds (wider ranges for tolerance) ──
        # Define color ranges (RGB thresholds)
        color_ranges = {
            'red': {'r_min': 170, 'r_max': 255, 'g_max': 120, 'b_max': 120},
            'blue': {'b_min': 120, 'b_max': 255, 'r_max': 80, 'g_max': 100},
            'green': {'g_min': 150, 'g_max': 255, 'r_max': 130, 'b_max': 130},
            'yellow': {'r_min': 180, 'g_min': 180, 'b_max': 120},
            'orange': {'r_min': 180, 'g_min': 80, 'g_max': 200, 'b_max': 80},
            'purple': {'r_min': 120, 'b_min': 120, 'g_max': 120},
            'gray': {'r_min': 80, 'r_max': 210, 'g_min': 80, 'g_max': 210, 'b_min': 80, 'b_max': 210},
            'black': {'r_max': 80, 'g_max': 80, 'b_max': 80},
            'magenta': {'r_min': 160, 'r_max': 255, 'g_max': 120, 'b_min': 160, 'b_max': 255},
            'pink': {'r_min': 180, 'r_max': 255, 'g_max': 170, 'b_min': 130, 'b_max': 255},
            'light blue': {'r_min': 100, 'r_max': 200, 'g_min': 100, 'g_max': 200, 'b_min': 200, 'b_max': 255},
            'cyan': {'r_max': 120, 'g_min': 160, 'g_max': 255, 'b_min': 180, 'b_max': 255},
            'light green': {'r_min': 80, 'r_max': 210, 'g_min': 180, 'g_max': 255, 'b_max': 170},
            'dark blue': {'r_max': 70, 'g_max': 70, 'b_min': 120, 'b_max': 255},
            'dark red': {'r_min': 120, 'r_max': 210, 'g_max': 70, 'b_max': 70},
            'brown': {'r_min': 100, 'r_max': 210, 'g_min': 40, 'g_max': 140, 'b_max': 100},
            'teal': {'r_max': 100, 'g_min': 110, 'g_max': 210, 'b_min': 110, 'b_max': 210},
        }
        
        target_color = target_color_name.lower()
        
        if target_color not in color_ranges:
            # Fallback: highlight any non-white, non-background pixels
            return self._extract_non_white_pixels(img_array)
        
        color_range = color_ranges[target_color]
        
        # Extract pixels matching color range (vectorized)
        mask = self._vectorized_color_mask(img_array, color_range)
        return self._mask_to_coords(mask)
    
    def _pixel_matches_color(self, r: int, g: int, b: int, color_range: Dict) -> bool:
        """Check if RGB values match color range."""
        r_match = (color_range.get('r_min', 0) <= r <= color_range.get('r_max', 255))
        g_match = (color_range.get('g_min', 0) <= g <= color_range.get('g_max', 255))
        b_match = (color_range.get('b_min', 0) <= b <= color_range.get('b_max', 255))
        
        return r_match and g_match and b_match
    
    def _vectorized_color_mask(self, img_array: np.ndarray, color_range: Dict) -> np.ndarray:
        """
        Build a boolean mask for all pixels matching a colour range (vectorized).
        
        Args:
            img_array: numpy array of shape (H, W, 3+)
            color_range: Dict with r_min/r_max/g_min/g_max/b_min/b_max keys
            
        Returns:
            Boolean mask of shape (H, W)
        """
        r = img_array[:, :, 0].astype(np.int16)
        g = img_array[:, :, 1].astype(np.int16)
        b = img_array[:, :, 2].astype(np.int16)
        
        mask = (
            (r >= color_range.get('r_min', 0)) & (r <= color_range.get('r_max', 255)) &
            (g >= color_range.get('g_min', 0)) & (g <= color_range.get('g_max', 255)) &
            (b >= color_range.get('b_min', 0)) & (b <= color_range.get('b_max', 255))
        )
        return mask
    
    @staticmethod
    def _mask_to_coords(mask: np.ndarray) -> List[Tuple[int, int]]:
        """Convert a 2-D boolean mask to a list of (x, y) pixel coordinates."""
        ys, xs = np.where(mask)
        return list(zip(xs.tolist(), ys.tolist()))

    # ─────────────────────────────────────────────────────────────
    #  HSV-based colour extraction  (cross-machine robust)
    # ─────────────────────────────────────────────────────────────

    _HSV_RANGES: Dict[str, List[Tuple[Tuple[int,int,int], Tuple[int,int,int]]]] = {
        # Each entry is a list of (lower, upper) HSV tuples.
        # H is 0-179 (PIL/numpy convention: 0-360 scaled to 0-179)
        # S is 0-255,  V is 0-255.
        'red':        [((0, 70, 80), (10, 255, 255)),
                       ((170, 70, 80), (179, 255, 255))],     # red wraps around hue 0
        'blue':       [((100, 70, 50), (130, 255, 255))],
        'green':      [((35, 50, 50), (85, 255, 255))],
        'yellow':     [((20, 80, 80), (35, 255, 255))],
        'orange':     [((10, 100, 100), (22, 255, 255))],
        'purple':     [((125, 40, 40), (160, 255, 255))],
        'magenta':    [((145, 60, 80), (175, 255, 255))],
        'pink':       [((140, 30, 150), (175, 255, 255))],
        'cyan':       [((80, 70, 70), (100, 255, 255))],
        'teal':       [((80, 50, 50), (100, 255, 200))],
        'light blue': [((95, 40, 120), (115, 255, 255))],
        'dark blue':  [((100, 80, 30), (130, 255, 180))],
        'dark red':   [((0, 80, 40), (10, 255, 180)),
                       ((170, 80, 40), (179, 255, 180))],
        'light green':[((35, 30, 100), (85, 200, 255))],
        'brown':      [((8, 60, 30), (22, 255, 180))],
    }

    def _extract_via_hsv(self, img_rgb: np.ndarray,
                         color_name: str) -> Optional[List[Tuple[int, int]]]:
        """Extract pixels for *color_name* using HSV ranges.

        Returns ``None`` if the colour name has no HSV definition
        (caller should fall back to RGB thresholds).

        This is more robust than pure RGB because HSV separates
        luminance from chrominance — JPEG artifacts, anti-aliasing,
        and monitor gamma differences mainly affect V, not H/S.
        """
        key = color_name.strip().lower()
        ranges = self._HSV_RANGES.get(key)
        if ranges is None:
            return None

        # Convert RGB → HSV (PIL: H 0-179, S 0-255, V 0-255)
        hsv = np.array(Image.fromarray(img_rgb).convert('HSV'))

        combined_mask = np.zeros(hsv.shape[:2], dtype=bool)
        for lo, hi in ranges:
            lo_arr = np.array(lo, dtype=np.uint8)
            hi_arr = np.array(hi, dtype=np.uint8)
            mask = (
                (hsv[:, :, 0] >= lo_arr[0]) & (hsv[:, :, 0] <= hi_arr[0]) &
                (hsv[:, :, 1] >= lo_arr[1]) & (hsv[:, :, 1] <= hi_arr[1]) &
                (hsv[:, :, 2] >= lo_arr[2]) & (hsv[:, :, 2] <= hi_arr[2])
            )
            combined_mask |= mask

        return self._mask_to_coords(combined_mask)

    @staticmethod
    def _pixels_to_debug_mask(pixels: List[Tuple[int, int]],
                              shape: Tuple[int, ...]) -> np.ndarray:
        """Create a binary debug image from a pixel list."""
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for x, y in pixels:
            if 0 <= y < h and 0 <= x < w:
                mask[y, x] = 255
        return mask

    def _calculate_dynamic_color_range(self, rgb_samples: List[Tuple[int, int, int]], 
                                       margin_std: float = 2.5) -> Dict[str, int]:
        """
        Calculate dynamic RGB range from sampled pixels.
        
        Args:
            rgb_samples: List of (r, g, b) tuples sampled from the curve
            margin_std: Number of standard deviations to use as margin (default 1.5)
            
        Returns:
            Dictionary with dynamic color range (r_min, r_max, g_min, g_max, b_min, b_max)
        """
        if len(rgb_samples) < 2:
            return {}
        
        rgb_array = np.array(rgb_samples)
        r_vals = rgb_array[:, 0]
        g_vals = rgb_array[:, 1]
        b_vals = rgb_array[:, 2]
        
        # Calculate mean and std for each channel
        r_mean, r_std = np.mean(r_vals), np.std(r_vals)
        g_mean, g_std = np.mean(g_vals), np.std(g_vals)
        b_mean, b_std = np.mean(b_vals), np.std(b_vals)
        
        # Create ranges with margin (generous minimum for JPEG tolerance)
        margin_r = max(r_std * margin_std, 15)
        margin_g = max(g_std * margin_std, 15)
        margin_b = max(b_std * margin_std, 15)
        
        return {
            'r_min': max(0, int(r_mean - margin_r)),
            'r_max': min(255, int(r_mean + margin_r)),
            'g_min': max(0, int(g_mean - margin_g)),
            'g_max': min(255, int(g_mean + margin_g)),
            'b_min': max(0, int(b_mean - margin_b)),
            'b_max': min(255, int(b_mean + margin_b)),
        }
    
    def extract_color_pixels_dynamic(self, image: Image.Image, target_color_name: str) -> List[Tuple[int, int]]:
        """
        Extract pixel coordinates using dynamic color range adaptation.
        
        Two-pass extraction:
        1. Initial pass with hardcoded range to find approximate curve pixels
        2. Calculate dynamic range from those samples
        3. Final pass with dynamic range for accurate extraction
        
        Args:
            image: PIL Image object
            target_color_name: Color name (e.g., 'red', 'blue', 'green')
            
        Returns:
            List of (x, y) pixel coordinates matching the color
        """
        img_array = np.array(image)
        target_color = target_color_name.lower()
        
        # Hardcoded color ranges for initial pass (same widened ranges as extract_color_pixels)
        color_ranges = {
            'red': {'r_min': 170, 'r_max': 255, 'g_max': 120, 'b_max': 120},
            'blue': {'b_min': 120, 'b_max': 255, 'r_max': 80, 'g_max': 100},
            'green': {'g_min': 150, 'g_max': 255, 'r_max': 130, 'b_max': 130},
            'yellow': {'r_min': 180, 'g_min': 180, 'b_max': 120},
            'orange': {'r_min': 180, 'g_min': 80, 'g_max': 200, 'b_max': 80},
            'purple': {'r_min': 120, 'b_min': 120, 'g_max': 120},
            'gray': {'r_min': 80, 'r_max': 210, 'g_min': 80, 'g_max': 210, 'b_min': 80, 'b_max': 210},
            'black': {'r_max': 80, 'g_max': 80, 'b_max': 80},
            'magenta': {'r_min': 160, 'r_max': 255, 'g_max': 120, 'b_min': 160, 'b_max': 255},
            'pink': {'r_min': 180, 'r_max': 255, 'g_max': 170, 'b_min': 130, 'b_max': 255},
            'light blue': {'r_min': 100, 'r_max': 200, 'g_min': 100, 'g_max': 200, 'b_min': 200, 'b_max': 255},
            'cyan': {'r_max': 120, 'g_min': 160, 'g_max': 255, 'b_min': 180, 'b_max': 255},
            'light green': {'r_min': 80, 'r_max': 210, 'g_min': 180, 'g_max': 255, 'b_max': 170},
            'dark blue': {'r_max': 70, 'g_max': 70, 'b_min': 120, 'b_max': 255},
            'dark red': {'r_min': 120, 'r_max': 210, 'g_max': 70, 'b_max': 70},
            'brown': {'r_min': 100, 'r_max': 210, 'g_min': 40, 'g_max': 140, 'b_max': 100},
            'teal': {'r_max': 100, 'g_min': 110, 'g_max': 210, 'b_min': 110, 'b_max': 210},
        }
        
        if target_color not in color_ranges:
            # Unknown color: fallback to non-white extraction
            return self._extract_non_white_pixels(img_array)
        
        # ─── PASS 1: Initial extraction with hardcoded range (vectorized) ───
        initial_range = color_ranges[target_color]
        initial_mask = self._vectorized_color_mask(img_array, initial_range)
        initial_pixels = self._mask_to_coords(initial_mask)
        
        # If we don't have enough initial samples, return what we have
        if len(initial_pixels) < 10:
            return initial_pixels
        
        # ─── Extract RGB samples from initial pixels ───
        rgb_samples = [tuple(img_array[y, x, :3]) for x, y in initial_pixels]
        
        # ─── Calculate dynamic range from samples ───
        dynamic_range = self._calculate_dynamic_color_range(rgb_samples)
        
        if not dynamic_range:
            # If dynamic calculation failed, return initial extraction
            return initial_pixels
        
        # ─── PASS 2: Final extraction with dynamic range (vectorized) ───
        # Union initial + dynamic masks to avoid losing pixels from Pass 1
        final_mask = initial_mask | self._vectorized_color_mask(img_array, dynamic_range)
        _save_debug_image(f"dynamic_{target_color}", final_mask)
        final_pixels = self._mask_to_coords(final_mask)
        
        return final_pixels if final_pixels else initial_pixels
    
    def _extract_non_white_pixels(self, img_array: np.ndarray) -> List[Tuple[int, int]]:
        """Extract non-white, non-background pixels (vectorized)."""
        brightness = np.mean(img_array[:, :, :3].astype(np.float32), axis=2)
        mask = brightness < 240
        return self._mask_to_coords(mask)
    
    def filter_spatially_connected(self, pixels: List[Tuple[int, int]],
                                    image_width: int, image_height: int
                                    ) -> List[Tuple[int, int]]:
        """
        Keep only the largest spatially connected band of pixels.
        
        Paints extracted pixels onto a binary mask, runs connected-component
        labelling, and returns only the pixels belonging to the largest
        component.  This strips isolated blobs caused by colour overlap
        between adjacent curves (e.g. dark-blue pixels leaking into purple).
        
        Uses 8-connectivity so diagonally adjacent pixels count as connected.
        
        Args:
            pixels: List of (x, y) pixel coordinates
            image_width: Full image width
            image_height: Full image height
            
        Returns:
            Filtered list of (x, y) pixel coordinates
        """
        if len(pixels) < 10:
            return pixels
        
        from scipy.ndimage import label as ndimage_label
        
        # Paint pixels onto a binary mask
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        for x, y in pixels:
            if 0 <= y < image_height and 0 <= x < image_width:
                mask[y, x] = 1
        
        # 8-connectivity kernel
        structure = np.ones((3, 3), dtype=int)
        labelled, n_components = ndimage_label(mask, structure=structure)
        
        if n_components <= 1:
            return pixels
        
        # Keep ALL components that span enough of the image width
        # (real curves are wide; stray colour blobs are narrow)
        min_hspan = max(10, int(image_width * 0.05))  # >= 5% of width
        component_sizes = np.bincount(labelled.ravel())
        component_sizes[0] = 0  # ignore background

        keep_labels = set()
        for comp_id in range(1, n_components + 1):
            if component_sizes[comp_id] < 5:
                continue
            ys_c, xs_c = np.where(labelled == comp_id)
            hspan = int(xs_c.max()) - int(xs_c.min()) + 1
            if hspan >= min_hspan:
                keep_labels.add(comp_id)

        # Fallback: if nothing passes the span test, keep just the largest
        if not keep_labels:
            largest_label = int(np.argmax(component_sizes))
            keep_labels = {largest_label}

        filtered = [(x, y) for x, y in pixels
                    if labelled[y, x] in keep_labels]
        
        return filtered if len(filtered) >= 5 else pixels

    def filter_spatially_near_anchor(
        self,
        pixels: List[Tuple[int, int]],
        image_width: int,
        image_height: int,
        anchor_start: Tuple[int, int],
        anchor_end: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Keep only the connected component(s) closest to an anchor pair.

        Paints *pixels* onto a binary mask, runs 8-connectivity labelling,
        then scores each component by the minimum Euclidean distance from
        the anchor start **or** end point to any pixel in that component.
        Only components that contain (or are closest to) at least one
        anchor are kept.

        This lets the user resolve ambiguity when multiple blobs of the
        same colour exist — the anchor selects the correct blob.

        Falls back to :meth:`filter_spatially_connected` when neither
        anchor is near any component.
        """
        if len(pixels) < 10:
            return pixels

        from scipy.ndimage import label as ndimage_label

        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        for x, y in pixels:
            if 0 <= y < image_height and 0 <= x < image_width:
                mask[y, x] = 1

        structure = np.ones((3, 3), dtype=int)
        labelled, n_components = ndimage_label(mask, structure=structure)

        if n_components <= 1:
            return pixels

        # For each component, compute min distance to either anchor
        sx, sy = anchor_start
        ex, ey = anchor_end
        keep_labels: set = set()
        best_dist = float('inf')
        best_label = -1

        for comp_id in range(1, n_components + 1):
            ys_c, xs_c = np.where(labelled == comp_id)
            if len(ys_c) < 5:
                continue
            # Vectorized min distance to start anchor
            d_start = float(np.min((xs_c - sx) ** 2 + (ys_c - sy) ** 2))
            # Vectorized min distance to end anchor
            d_end = float(np.min((xs_c - ex) ** 2 + (ys_c - ey) ** 2))
            d_min = min(d_start, d_end)

            # If anchor lands inside or very close (<15 px) to this component
            if d_min < 15 ** 2:
                keep_labels.add(comp_id)

            if d_min < best_dist:
                best_dist = d_min
                best_label = comp_id

        # Fallback: if no component is close enough, keep the nearest one
        if not keep_labels and best_label > 0:
            keep_labels = {best_label}

        if not keep_labels:
            return self.filter_spatially_connected(pixels, image_width, image_height)

        filtered = [(x, y) for x, y in pixels
                    if labelled[y, x] in keep_labels]

        return filtered if len(filtered) >= 5 else pixels
    
    # ─────────────────────────────────────────────────────────────
    #  Grayscale / B&W image support
    # ─────────────────────────────────────────────────────────────
    
    def is_grayscale_image(self, image: Image.Image) -> bool:
        """
        Detect whether an image is grayscale (B&W / shades of gray).
        
        A pixel is considered grayscale if |R-G|, |R-B|, and |G-B| are all
        within a tight tolerance.  The image is grayscale when >= 85% of
        non-background pixels satisfy that condition.
        """
        img_array = np.array(image).astype(np.float32)
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        brightness = (r + g + b) / 3.0
        
        # Only test pixels that aren't background (white) or axes (black)
        mask = (brightness > 25) & (brightness < 240)
        if np.sum(mask) < 50:
            return False
        
        max_diff = np.maximum(
            np.maximum(np.abs(r[mask] - g[mask]), np.abs(r[mask] - b[mask])),
            np.abs(g[mask] - b[mask])
        )
        grayscale_ratio = float(np.sum(max_diff <= 12) / len(max_diff))
        return bool(grayscale_ratio >= 0.85)
    
    def extract_curves_grayscale(self, image: Image.Image, num_curves: int,
                                  plot_area: Tuple[int, int, int, int]
                                  ) -> Dict[int, List[Tuple[int, int]]]:
        """
        Extract curves from a grayscale / B&W image.

        Robust pipeline (cross-machine deterministic):
        1. Crop to plot area, convert to true grayscale (uint8).
        2. Denoise with a small median filter (removes JPEG noise).
        3. Adaptive Otsu threshold → binary mask of dark strokes.
        4. Morphological close (connect broken dashes / thin lines).
        5. Suppress horizontal/vertical grid lines (projection filter).
        6. Column-scan tracker: for each column, find vertical runs of
           dark pixels and track their centroids left-to-right.
        7. Filter tracks by horizontal extent (real curves span the plot).
        8. Sort by mean y (topmost = index 0).

        Column-scan tracking is superior to connected-component labelling
        because it correctly separates curves that touch or cross each
        other, which is common on real performance charts.

        This is robust to:
        - Different JPEG quality / compression artifacts across machines
        - Anti-aliasing differences (font rendering, line smoothing)
        - Monitor DPI / image scaling (adaptive, not pixel-count based)

        Args:
            image: PIL Image
            num_curves: Expected curve count (from LLM) — used as hint only
            plot_area: (left, top, right, bottom) px bounds

        Returns:
            Dict mapping curve index (0 = topmost) to list of (x, y) pixel coords
        """
        from scipy.ndimage import label as ndimage_label, binary_closing, binary_opening

        img_array = np.array(image)
        p_left, p_top, p_right, p_bottom = plot_area

        # Inset to skip axis-line pixels right at the boundary
        inset = 5
        p_left  = min(p_left + inset,  p_right - 1)
        p_top   = min(p_top + inset,   p_bottom - 1)
        p_right = max(p_right - inset,  p_left + 1)
        p_bottom = max(p_bottom - inset, p_top + 1)

        region = img_array[p_top:p_bottom, p_left:p_right]
        if region.ndim == 3:
            gray = np.mean(region[:, :, :3].astype(np.float32), axis=2).astype(np.uint8)
        else:
            gray = region.astype(np.uint8)
        region_h, region_w = gray.shape
        logger.debug("grayscale extraction: region %dx%d", region_w, region_h)

        # ── Step 1: Denoise (3×3 median — removes salt-and-pepper) ──
        gray_pil = Image.fromarray(gray).filter(ImageFilter.MedianFilter(size=3))
        gray = np.array(gray_pil)

        # ── Step 2: Adaptive Otsu threshold ──
        threshold = self._otsu_threshold(gray)
        # We want dark-on-light: pixels darker than threshold → True
        # Use <= and add a small margin to capture curves whose gray value
        # sits right at the Otsu boundary (common with anti-aliased lines).
        binary = gray <= min(threshold + 10, 200)
        _save_debug_image("gs_binary", binary.astype(np.uint8) * 255)

        # ── Step 3: Morphological close to connect dashes / thin strokes ──
        close_kernel = np.ones((3, 5), dtype=bool)   # wider than tall → connect along x
        binary = binary_closing(binary, structure=close_kernel, iterations=1)

        # ── Step 4: Suppress horizontal / vertical grid lines ──
        # A row is a grid line if >75% of its pixels are dark
        row_fill = binary.sum(axis=1) / region_w
        grid_rows = row_fill > 0.75
        binary[grid_rows, :] = False
        # Same for columns
        col_fill = binary.sum(axis=0) / region_h
        grid_cols = col_fill > 0.75
        binary[:, grid_cols] = False
        _save_debug_image("gs_after_grid_suppress", binary.astype(np.uint8) * 255)

        # Small opening to remove residual specks after grid removal
        open_kernel = np.ones((2, 2), dtype=bool)
        binary = binary_opening(binary, structure=open_kernel, iterations=1)

        # ── Step 5: Column-scan curve tracker ──
        # Instead of dilation + connected-component labelling (which merges
        # touching curves into a single blob), scan each column individually
        # and track vertical "runs" of dark pixels across the image width.
        # This correctly separates curves even where they touch or cross.
        from scipy.ndimage import binary_dilation

        # --- 5a: find dark-pixel runs in every column ---
        RUN_GAP = max(3, int(region_h * 0.005))     # min vertical gap to split runs
        column_runs: Dict[int, list] = {}            # x → [(centroid_y, run_top, run_bot), ...]
        for x in range(region_w):
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
            column_runs[x] = [(int((r[0] + r[1]) // 2), int(r[0]), int(r[1])) for r in runs]

        # --- 5b: slope-predicting nearest-neighbour tracker ---
        # Each "track" accumulates (x, centroid_y) pairs for one curve.
        # We predict the expected y in the next column using a short
        # slope history, which prevents the tracker from jumping to an
        # adjacent curve when two curves are close together.
        MAX_Y_JUMP = max(8, int(region_h * 0.03))   # max prediction error (tight)
        MAX_X_GAP  = max(20, int(region_w * 0.10))   # stale track threshold
        SLOPE_WINDOW = 5  # columns for slope estimation

        tracks: list = []          # list of lists: [[(x, cy), ...], ...]
        track_last_x: list = []    # last-seen x for each track
        track_last_y: list = []    # last-seen y for each track
        track_slope: list = []     # estimated dy/dx for each track

        def _estimate_slope(track_pts):
            """Estimate local slope from last few points."""
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

            # Build cost matrix using predicted y position
            used_tracks: set = set()
            used_cents: set = set()
            pairs = []
            for ci, cy in enumerate(centroids):
                for ti in range(len(tracks)):
                    x_gap = x - track_last_x[ti]
                    if x_gap > MAX_X_GAP:
                        continue
                    # Predict where the track should be at column x
                    predicted_y = track_last_y[ti] + track_slope[ti] * x_gap
                    dist = abs(cy - predicted_y)
                    if dist <= MAX_Y_JUMP:
                        pairs.append((dist, ci, ti))
            pairs.sort()  # best (smallest distance) first

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

        # --- 5c: post-track running-median smoothing ---
        # Remove jumps: for each point, if its y deviates more than a
        # threshold from the running median of its neighbours, drop it.
        SMOOTH_HALF = 5   # half-window for running median
        SMOOTH_THR  = max(6, int(region_h * 0.015))  # max deviation from local median
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

        # --- 5c: filter tracks ---
        min_track_width = max(10, int(region_w * 0.15))
        valid_tracks = []
        for t in tracks:
            xs_t = [p[0] for p in t]
            ys_t = [p[1] for p in t]
            x_span = max(xs_t) - min(xs_t)
            y_span = max(ys_t) - min(ys_t)

            # (i) Too short horizontally
            if x_span < min_track_width:
                continue

            # (ii) Steep / nearly-vertical lines (surge lines)
            slope = y_span / max(x_span, 1)
            if slope > 1.5:
                continue

            # (iii) Dashed / dotted lines: low column coverage
            unique_x = len(set(xs_t))
            coverage = unique_x / max(x_span, 1)
            if coverage < 0.50:
                continue

            valid_tracks.append(t)

        # --- 5d: rank and limit to num_curves ---
        # Score each track by (x_span * coverage) — prefer long, dense tracks.
        # Keep at most num_curves tracks (the LLM-reported curve count).
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
        logger.debug("grayscale extraction: %d tracks after column-scan",
                     len(valid_tracks))

        _save_debug_image("gs_binary_final", binary.astype(np.uint8) * 255)

        # ── Step 6: Convert tracked centroids to pixel coordinates ──
        result: Dict[int, List[Tuple[int, int]]] = {}
        for idx, track in enumerate(valid_tracks):
            # Each track point is (col_x, centroid_y) in region coords
            # Convert to full-image pixel coords
            thinned = [(x + p_left, y + p_top) for x, y in track]
            result[idx] = thinned

        # Debug: save colour-coded tracks
        if _DEBUG_DIR:
            _trk_img = np.zeros((*binary.shape, 3), dtype=np.uint8)
            _trk_colors = [(255,0,0),(0,180,0),(0,0,255),(255,165,0),(128,0,255),(0,200,200)]
            for i, track in enumerate(valid_tracks):
                c = _trk_colors[i % len(_trk_colors)]
                for tx, ty in track:
                    if 0 <= ty < _trk_img.shape[0] and 0 <= tx < _trk_img.shape[1]:
                        _trk_img[ty, tx] = c
            _save_debug_image("gs_tracks", _trk_img)

        return result

    @staticmethod
    def _otsu_threshold(gray: np.ndarray) -> int:
        """Compute Otsu's threshold for a uint8 grayscale image.

        Deterministic (no randomness), works identically on every machine
        regardless of numpy/scipy version.
        """
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

        # Clamp to reasonable range for chart images
        return max(40, min(best_t, 200))

    @staticmethod
    def _cluster_tick_indices(tick_indices: np.ndarray,
                              max_gap: int = 3) -> List[List[int]]:
        """Group adjacent indices into tick clusters (thick ticks span 2-4 px)."""
        clusters: List[List[int]] = []
        cur = [int(tick_indices[0])]
        for i in range(1, len(tick_indices)):
            if tick_indices[i] - tick_indices[i - 1] <= max_gap:
                cur.append(int(tick_indices[i]))
            else:
                clusters.append(cur)
                cur = [int(tick_indices[i])]
        clusters.append(cur)
        return clusters

    def _refine_plot_area_with_ticks(
        self,
        image: Image.Image,
        plot_area: Tuple[int, int, int, int],
    ) -> Tuple[int, int, int, int]:
        """Refine all four plot-area boundaries using axis tick marks.

        Y-axis ticks (horizontal dashes left of the y-axis) refine
        top/bottom; X-axis ticks (vertical dashes below the x-axis)
        refine left/right.  The tick positions correspond to labelled
        axis values, so bounding by the first and last tick gives the
        most accurate pixel↔axis mapping.

        Uses ``last_tick + 1`` for bottom/right so that the tick pixel
        itself is included in the numpy slice (exclusive upper bound).
        """
        img_array = np.array(image)
        if img_array.ndim == 3:
            gray = np.mean(img_array[:, :, :3], axis=2)
        else:
            gray = img_array.astype(float)

        height, width = gray.shape
        p_left, p_top, p_right, p_bottom = plot_area
        dark_threshold = 100

        # ── Y-axis tick detection (refines top / bottom) ──────────
        refined_top, refined_bottom = p_top, p_bottom

        tick_strip_w = min(20, p_left)
        strip_left = max(0, p_left - tick_strip_w)
        strip_right = p_left + 2  # include axis edge
        if strip_right > strip_left:
            strip = gray[:, strip_left:strip_right]
            dark_per_row = np.sum(strip < dark_threshold, axis=1)

            region_dark = dark_per_row[p_top:p_bottom]
            if len(region_dark) > 0:
                median_dark = float(np.median(region_dark))
                tick_threshold = median_dark + max(1, median_dark * 0.3)
                tick_rows_rel = np.where(region_dark >= tick_threshold)[0]

                if len(tick_rows_rel) >= 2:
                    y_clusters = self._cluster_tick_indices(tick_rows_rel)
                    if len(y_clusters) >= 2:
                        first_tick_y = int(np.mean(y_clusters[0])) + p_top
                        last_tick_y = int(np.mean(y_clusters[-1])) + p_top
                        tick_span = last_tick_y - first_tick_y
                        if tick_span >= (p_bottom - p_top) * 0.3:
                            # +1 on bottom so the tick row is INCLUDED
                            # in numpy slice [top:bottom)
                            cand_top = first_tick_y
                            cand_bot = last_tick_y + 1
                            if cand_top >= p_top - 10 and cand_bot <= p_bottom + 10:
                                refined_top = cand_top
                                refined_bottom = cand_bot

        # ── X-axis tick detection (refines left / right) ──────────
        refined_left, refined_right = p_left, p_right

        # Start the strip a few pixels BELOW the axis line so the
        # continuous dark axis row doesn't drown out the tick signal.
        # Cap depth at 12 px to avoid picking up axis-label text.
        tick_strip_h = min(12, height - p_bottom)
        strip_top_x = p_bottom + 2  # skip the axis line itself
        strip_bot_x = min(height, p_bottom + max(tick_strip_h, 8))
        # Widen column search beyond current boundaries so ticks just
        # outside the rough detect_plot_area can still be found.
        margin_x = max(50, int((p_right - p_left) * 0.08))
        search_left = max(0, p_left - margin_x)
        search_right = min(width, p_right + margin_x)
        # Use a stricter brightness cutoff for x-ticks: below the axis
        # line there is label text that passes the generous dark_threshold
        # (100) but tick marks are solid black (< 80).
        x_dark_thr = min(dark_threshold, 80)
        if strip_bot_x > strip_top_x and search_right > search_left:
            strip_x = gray[strip_top_x:strip_bot_x, :]
            dark_per_col = np.sum(strip_x < x_dark_thr, axis=0)

            region_dark_x = dark_per_col[search_left:search_right]
            if len(region_dark_x) > 0:
                median_dark_x = float(np.median(region_dark_x))
                tick_threshold_x = median_dark_x + max(1, median_dark_x * 0.3)
                tick_cols_rel = np.where(region_dark_x >= tick_threshold_x)[0]

                if len(tick_cols_rel) >= 2:
                    x_clusters = self._cluster_tick_indices(tick_cols_rel)
                    if len(x_clusters) >= 2:
                        # Compute cluster centers (absolute pixel coords)
                        centers = [int(np.mean(c)) + search_left
                                   for c in x_clusters]

                        # Periodicity filter: axis-corner artifacts
                        # produce tiny spacings that differ from the
                        # regular tick spacing.  Strip them from both
                        # ends while there are ≥ 3 clusters remaining.
                        if len(centers) >= 3:
                            spacings = [centers[i + 1] - centers[i]
                                        for i in range(len(centers) - 1)]
                            med_sp = float(np.median(spacings))
                            min_sp = med_sp * 0.5
                            while (len(centers) >= 3
                                   and centers[1] - centers[0] < min_sp):
                                centers.pop(0)
                            while (len(centers) >= 3
                                   and centers[-1] - centers[-2] < min_sp):
                                centers.pop()

                        if len(centers) >= 2:
                            first_tick_x = centers[0]
                            last_tick_x = centers[-1]
                            tick_span_x = last_tick_x - first_tick_x
                            if tick_span_x >= (p_right - p_left) * 0.3:
                                cand_left = first_tick_x
                                cand_right = last_tick_x + 1
                                if (cand_left >= p_left - margin_x
                                        and cand_right <= p_right + margin_x):
                                    refined_left = cand_left
                                    refined_right = cand_right

        logger.debug(
            "tick_refine: (%d,%d,%d,%d) → (%d,%d,%d,%d)",
            p_left, p_top, p_right, p_bottom,
            refined_left, refined_top, refined_right, refined_bottom,
        )
        return (refined_left, refined_top, refined_right, refined_bottom)
    
    def detect_plot_area(self, image: Image.Image, dark_threshold: int = 80,
                         line_ratio: float = 0.3) -> Tuple[int, int, int, int]:
        """
        Detect the actual plot/chart area boundaries in pixel coordinates.
        
        Finds axis lines (dark horizontal/vertical lines) to determine where
        the plot region starts and ends, excluding labels, titles, and margins.
        
        Args:
            image: PIL Image object
            dark_threshold: Max brightness to consider a pixel as 'dark' (axis line)
            line_ratio: Min fraction of row/col that must be dark to count as axis line
            
        Returns:
            Tuple (plot_left, plot_top, plot_right, plot_bottom) in pixel coordinates
        """
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Convert to grayscale brightness
        if img_array.ndim == 3:
            gray = np.mean(img_array[:, :, :3], axis=2)
        else:
            gray = img_array.astype(float)
        
        # Create dark pixel mask
        dark_mask = gray < dark_threshold
        
        # ─── Detect horizontal axis lines ───
        # Count dark pixels per row
        dark_per_row = np.sum(dark_mask, axis=1)
        min_dark_h = int(width * line_ratio)
        horizontal_line_rows = np.where(dark_per_row >= min_dark_h)[0]
        
        # ─── Detect vertical axis lines ───
        # Count dark pixels per column
        dark_per_col = np.sum(dark_mask, axis=0)
        min_dark_v = int(height * line_ratio)
        vertical_line_cols = np.where(dark_per_col >= min_dark_v)[0]
        
        # ─── Reject dashed / reference lines ───
        # Solid axis lines have a high fill-ratio within their span;
        # dashed lines (surge, reference) have alternating dark/gap segments.
        def _is_solid(dark_1d: np.ndarray) -> bool:
            idx = np.where(dark_1d)[0]
            if len(idx) < 5:
                return False
            span = idx[-1] - idx[0] + 1
            return len(idx) / span > 0.70   # >70% filled = solid
        
        vertical_line_cols = np.array(
            [c for c in vertical_line_cols if _is_solid(dark_mask[:, c])],
            dtype=int,
        )
        horizontal_line_rows = np.array(
            [r for r in horizontal_line_rows if _is_solid(dark_mask[r, :])],
            dtype=int,
        )
        
        # ─── Determine plot boundaries ───
        # Default: use 10% margins as fallback
        plot_left = int(width * 0.10)
        plot_right = int(width * 0.90)
        plot_top = int(height * 0.05)
        plot_bottom = int(height * 0.85)
        
        if len(vertical_line_cols) > 0:
            # Left axis = leftmost cluster of vertical lines
            # Right boundary = rightmost vertical line (if box plot) or use right margin
            left_candidates = vertical_line_cols[vertical_line_cols < width // 2]
            right_candidates = vertical_line_cols[vertical_line_cols > width // 2]
            
            if len(left_candidates) > 0:
                plot_left = int(np.max(left_candidates))  # rightmost edge of left axis line cluster
            if len(right_candidates) > 0:
                plot_right = int(np.min(right_candidates))  # leftmost edge of right boundary
        
        if len(horizontal_line_rows) > 0:
            # Bottom axis = bottommost cluster of horizontal lines in lower half
            # Top boundary = topmost horizontal line (if box plot) or use top margin
            bottom_candidates = horizontal_line_rows[horizontal_line_rows > height // 2]
            top_candidates = horizontal_line_rows[horizontal_line_rows < height // 2]
            
            if len(bottom_candidates) > 0:
                plot_bottom = int(np.min(bottom_candidates))  # topmost edge of bottom axis line cluster
            if len(top_candidates) > 0:
                plot_top = int(np.max(top_candidates))  # bottommost edge of top boundary
        
        # Sanity checks: ensure reasonable boundaries
        if plot_right - plot_left < width * 0.2:
            plot_left = int(width * 0.10)
            plot_right = int(width * 0.90)
        if plot_bottom - plot_top < height * 0.2:
            plot_top = int(height * 0.05)
            plot_bottom = int(height * 0.85)
        
        return (plot_left, plot_top, plot_right, plot_bottom)
    
    def normalize_to_axis(self, pixel_coords: List[Tuple[int, int]], 
                         image_width: int, image_height: int,
                         plot_area: Optional[Tuple[int, int, int, int]] = None) -> List[Tuple[float, float]]:
        """
        Normalize pixel coordinates to axis coordinates.
        
        Uses fence-post mapping: the first extractable pixel (p_left, p_top)
        maps to (xMin, yMax) and the last extractable pixel
        (p_right-1, p_bottom-1) maps to (xMax, yMin).
        
        This fixes the +1-unit Y-shift that occurred when using
        p_right-p_left (N intervals) instead of p_right-p_left-1
        (N-1 intervals between N pixel positions).
        
        Args:
            pixel_coords: List of (pixel_x, pixel_y) tuples
            image_width: Image width in pixels
            image_height: Image height in pixels
            plot_area: Optional (left, top, right, bottom) pixel bounds of actual plot region.
                       If provided, mapping is relative to this region instead of full image.
            
        Returns:
            List of (axis_x, axis_y) tuples normalized to axis bounds
        """
        normalized = []
        
        # Use plot area boundaries if available, else full image
        if plot_area:
            p_left, p_top, p_right, p_bottom = plot_area
        else:
            p_left, p_top = 0, 0
            p_right, p_bottom = image_width, image_height
        
        # Fence-post: N pixels span N-1 intervals.
        # p_left..p_right-1 are extractable pixel positions (p_right is exclusive).
        # Number of intervals = (p_right - 1) - p_left = p_right - p_left - 1.
        p_width  = max(p_right - p_left - 1, 1)
        p_height = max(p_bottom - p_top - 1, 1)
        
        for px, py in pixel_coords:
            # Normalize pixel relative to plot area to 0-1 range
            norm_x = float(px - p_left) / p_width
            norm_y = 1.0 - (float(py - p_top) / p_height)  # Flip Y
            
            # Clamp to [0, 1] — pixels outside plot area get clamped
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))
            
            # Map to axis bounds
            axis_x = self.xMin + norm_x * (self.xMax - self.xMin)
            axis_y = self.yMin + norm_y * (self.yMax - self.yMin)
            
            normalized.append((axis_x, axis_y))
        
        return normalized
    
    def clean_coordinates_ransac(self, coordinates: List[Tuple[float, float]], 
                                threshold: float = 0.1) -> List[Tuple[float, float]]:
        """
        Remove noise from coordinates using RANSAC.
        
        Args:
            coordinates: List of (x, y) tuples
            threshold: RANSAC threshold for outlier detection
            
        Returns:
            List of cleaned (x, y) tuples
        """
        if len(coordinates) < 3:
            return coordinates
        
        coords_array = np.array(coordinates)
        X = coords_array[:, 0:1]  # x values as column vector
        y = coords_array[:, 1]    # y values as 1D array
        
        # Fit polynomial (degree 2) using RANSAC
        ransac = RANSACRegressor(
            estimator=PolynomialFeatures(degree=2),
            random_state=42,
            min_samples=3,
            residual_threshold=threshold * (self.yMax - self.yMin)
        )
        
        try:
            # Use a simple polynomial fitter instead
            # RANSAC with polynomial features requires wrapping
            inlier_mask = self._ransac_filter(X, y, threshold)
            cleaned = coords_array[inlier_mask]
            return [tuple(point) for point in cleaned]
        except Exception as e:
            print(f"RANSAC filtering failed: {e}. Returning original coordinates.")
            return coordinates
    
    def _ransac_filter(self, X: np.ndarray, y: np.ndarray, threshold: float) -> np.ndarray:
        """Simple RANSAC-like filtering."""
        # Fit a polynomial to all points
        try:
            coeffs = np.polyfit(X.flatten(), y, 2)
            poly = np.poly1d(coeffs)
            
            # Calculate residuals
            predictions = poly(X.flatten())
            residuals = np.abs(y - predictions)
            
            # Filter outliers
            threshold_value = threshold * (self.yMax - self.yMin)
            inlier_mask = residuals <= threshold_value
            
            # Ensure we keep at least 50% of points
            if np.sum(inlier_mask) < len(y) * 0.5:
                # Use percentile-based filtering
                percentile = 50
                threshold_value = np.percentile(residuals, percentile)
                inlier_mask = residuals <= threshold_value
            
            return inlier_mask
        except Exception:
            # If fitting fails, return all points
            return np.ones(len(y), dtype=bool)

    # ─────────────────────────────────────────────────────────────
    #  Local deviation cleaner  (shape-preserving)
    # ─────────────────────────────────────────────────────────────

    def clean_coordinates_local(self, coordinates: List[Tuple[float, float]],
                                window: int = 7,
                                sigma: float = 2.5) -> List[Tuple[float, float]]:
        """Remove outlier points using a local moving-window deviation filter.

        Unlike ``clean_coordinates_ransac`` (which fits a single global
        polynomial), this method evaluates each point against its local
        neighbourhood.  This preserves curves with multiple inflection
        points, S-shapes, or other non-polynomial forms.
        """
        if len(coordinates) < window:
            return coordinates

        arr = np.array(sorted(coordinates, key=lambda p: p[0]))
        xs, ys = arr[:, 0], arr[:, 1]
        n = len(ys)
        keep = np.ones(n, dtype=bool)
        half = window // 2

        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            local_y = ys[lo:hi]
            med = np.median(local_y)
            mad = np.median(np.abs(local_y - med))
            if mad < 1e-9:
                mad = np.std(local_y)
            if mad < 1e-9:
                continue
            if abs(ys[i] - med) > sigma * mad:
                keep[i] = False

        cleaned = arr[keep]
        if len(cleaned) < len(arr) * 0.6:
            return coordinates
        return [tuple(p) for p in cleaned]

    # ─────────────────────────────────────────────────────────────
    #  Iterative sigma-clip (fit-based outlier rejection)
    # ─────────────────────────────────────────────────────────────

    def _iterative_sigma_clip(self, coordinates: List[Tuple[float, float]],
                              degree: int = 2,
                              sigma: float = 2.5,
                              max_iter: int = 3) -> List[Tuple[float, float]]:
        """Iteratively fit a polynomial and reject outliers.

        Each round:
        1. Fit polynomial of given *degree* to the remaining points.
        2. Compute residuals.
        3. Remove points whose |residual| > sigma * MAD(residuals).
        4. Stop when no points are removed or *max_iter* reached.

        Preserves at least 60 % of original points.
        """
        if len(coordinates) < degree + 2:
            return coordinates

        arr = np.array(sorted(coordinates, key=lambda p: p[0]))
        for _ in range(max_iter):
            xs, ys = arr[:, 0], arr[:, 1]
            try:
                coeffs = np.polyfit(xs, ys, degree)
            except Exception:
                break
            poly = np.poly1d(coeffs)
            residuals = np.abs(ys - poly(xs))
            mad = np.median(residuals)
            if mad < 1e-9:
                break  # fit is already nearly perfect
            threshold = sigma * mad
            keep = residuals <= threshold
            # Never drop below 40 % of the original input
            if np.sum(keep) < len(coordinates) * 0.4:
                break
            if np.all(keep):
                break  # nothing to remove
            arr = arr[keep]

        return [tuple(p) for p in arr]

    # ─────────────────────────────────────────────────────────────
    #  Cubic smoothing spline  (any curve shape)
    # ─────────────────────────────────────────────────────────────

    def fit_spline_curve(self, coordinates: List[Tuple[float, float]],
                         n_output: int = 300,
                         smoothing: float = 0.0) -> Dict[str, Any]:
        """Fit a cubic smoothing spline to coordinate data.

        Unlike ``fit_polynomial_curve`` this can represent **any** curve
        shape (S-curves, multiple peaks, steep drops, etc.).
        """
        if len(coordinates) < 4:
            return {
                'degree': 'spline',
                'coefficients': None,
                'error': f'Not enough points ({len(coordinates)}) for spline fit'
            }

        arr = np.array(sorted(coordinates, key=lambda p: p[0]))
        xs, ys = arr[:, 0], arr[:, 1]

        # Remove duplicate x values (average y)
        ux, idx_inv = np.unique(xs, return_inverse=True)
        uy = np.zeros_like(ux)
        for i in range(len(ux)):
            uy[i] = np.mean(ys[idx_inv == i])

        if len(ux) < 4:
            return {
                'degree': 'spline',
                'coefficients': None,
                'error': 'Too few unique x values for spline'
            }

        try:
            from scipy.interpolate import UnivariateSpline
            # Smoothing factor: higher = smoother curve.
            # s ~ N means the spline is allowed total squared-error ≈ s.
            # Using N * var(y) * 0.05 gives a clean, smooth curve while
            # still following the data trend accurately.
            if smoothing > 0:
                s_val = smoothing
            else:
                y_var = float(np.var(uy)) if np.var(uy) > 1e-12 else 1.0
                s_val = len(ux) * y_var * 0.05
            k = min(3, len(ux) - 1)
            spl = UnivariateSpline(ux, uy, k=k, s=s_val)

            x_fit = np.linspace(float(ux[0]), float(ux[-1]), n_output)
            y_fit = spl(x_fit)

            y_pred = spl(ux)
            ss_res = float(np.sum((uy - y_pred) ** 2))
            ss_tot = float(np.sum((uy - np.mean(uy)) ** 2))
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0

            return {
                'degree': 'spline',
                'coefficients': None,
                'r_squared': float(r_squared),
                'fitted_points': [{'x': float(x), 'y': float(y)}
                                  for x, y in zip(x_fit, y_fit)],
                'original_point_count': len(coordinates),
                'equation': 'cubic smoothing spline'
            }
        except Exception as e:
            return {
                'degree': 'spline',
                'coefficients': None,
                'r_squared': None,
                'fitted_points': [{'x': float(x), 'y': float(y)}
                                  for x, y in zip(ux, uy)],
                'original_point_count': len(coordinates),
                'equation': f'spline fallback ({e})'
            }

    def fit_polynomial_curve(self, coordinates: List[Tuple[float, float]], 
                            degree: int = 2) -> Dict[str, Any]:
        """
        Fit a polynomial curve to coordinates.
        
        Args:
            coordinates: List of (x, y) tuples
            degree: Polynomial degree (default 2 for quadratic)
            
        Returns:
            Dictionary with coefficients and fitted points
        """
        if len(coordinates) < degree + 1:
            return {
                'degree': degree,
                'coefficients': None,
                'error': f'Not enough points ({len(coordinates)}) for polynomial degree {degree}'
            }
        
        coords_array = np.array(coordinates)
        X = coords_array[:, 0]
        y = coords_array[:, 1]
        
        try:
            # Fit polynomial
            coeffs = np.polyfit(X, y, degree)
            poly = np.poly1d(coeffs)
            
            # Calculate fitted points (300 for smooth rendering)
            x_min, x_max = X.min(), X.max()
            x_fit = np.linspace(x_min, x_max, 300)
            y_fit = poly(x_fit)
            
            # Calculate R-squared
            y_pred = poly(X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'degree': degree,
                'coefficients': coeffs.tolist(),
                'r_squared': float(r_squared),
                'fitted_points': [{'x': float(x), 'y': float(y)} for x, y in zip(x_fit, y_fit)],
                'original_point_count': len(coordinates),
                'equation': self._poly_equation_string(coeffs)
            }
        except Exception as e:
            return {
                'degree': degree,
                'coefficients': None,
                'error': str(e)
            }
    
    def calculate_curve_metrics(self, cleaned_coords: List[Tuple[float, float]],
                                fit_result: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate quality metrics comparing original extracted points to the fitted curve.
        
        Metrics:
        - delta_value: Mean absolute error between extracted points and fitted curve
        - delta_norm: Delta normalized by axis Y-range (scale-independent, 0-1)
        - iou: Intersection over Union of point bands (fitted vs extracted)
        - precision: Fraction of fitted curve region covered by actual data
        - recall: Fraction of actual data captured by the fitted curve region
        - delta_p95: 95th percentile of absolute errors (worst-case excluding outliers)
        
        Args:
            cleaned_coords: Cleaned (x, y) coordinate tuples from pixel extraction
            fit_result: Dictionary from fit_polynomial_curve with coefficients etc.
            
        Returns:
            Dictionary of metric names to values
        """
        metrics = {
            'delta_value': None,
            'delta_norm': None,
            'iou': None,
            'precision': None,
            'recall': None,
            'delta_p95': None,
        }
        
        if len(cleaned_coords) < 3:
            return metrics

        coeffs = fit_result.get('coefficients')
        fitted_pts = fit_result.get('fitted_points', [])

        coords_array = np.array(cleaned_coords)
        x_actual = coords_array[:, 0]
        y_actual = coords_array[:, 1]

        # Build an interpolator from fitted_points (works for BOTH
        # polynomial and spline results).
        if fitted_pts and len(fitted_pts) >= 2:
            fp_x = np.array([p['x'] for p in fitted_pts])
            fp_y = np.array([p['y'] for p in fitted_pts])
            y_fitted = np.interp(x_actual, fp_x, fp_y)
        elif coeffs is not None:
            poly = np.poly1d(coeffs)
            y_fitted = poly(x_actual)
        else:
            return metrics
        
        # ─── Delta Value (Mean Absolute Error) ───
        absolute_errors = np.abs(y_actual - y_fitted)
        delta_value = float(np.mean(absolute_errors))
        
        # ─── Delta Norm (Normalized MAE) ───
        y_range = self.yMax - self.yMin
        delta_norm = float(delta_value / y_range) if y_range > 0 else 0.0
        
        # ─── Delta P95 (95th percentile error) ───
        delta_p95 = float(np.percentile(absolute_errors, 95))
        
        # ─── IoU, Precision, Recall ───
        # Compare by binning x-values and checking y-overlap in each bin
        n_bins = 50
        x_min, x_max = float(np.min(x_actual)), float(np.max(x_actual))
        if x_max <= x_min:
            x_max = x_min + 1e-6
        
        bin_edges = np.linspace(x_min, x_max, n_bins + 1)
        
        # Band tolerance: how close y values need to be to count as "overlapping"
        band_tolerance = y_range * 0.02  # 2% of axis range
        
        intersection_count = 0
        actual_bins_occupied = 0
        fitted_bins_occupied = 0
        
        for i in range(n_bins):
            b_lo, b_hi = bin_edges[i], bin_edges[i + 1]
            
            # Actual points in this bin
            mask = (x_actual >= b_lo) & (x_actual < b_hi)
            actual_in_bin = y_actual[mask]
            
            # Fitted value at bin center (works for any fit type)
            x_center = (b_lo + b_hi) / 2.0
            if fitted_pts and len(fitted_pts) >= 2:
                y_fit_val = float(np.interp(x_center, fp_x, fp_y))
            else:
                y_fit_val = float(np.poly1d(coeffs)(x_center))
            
            has_actual = len(actual_in_bin) > 0
            has_fitted = True  # polynomial is defined everywhere
            
            if has_actual:
                actual_bins_occupied += 1
            if has_fitted:
                fitted_bins_occupied += 1
            
            # Check overlap: any actual point within tolerance of fitted value
            if has_actual and has_fitted:
                if np.any(np.abs(actual_in_bin - y_fit_val) <= band_tolerance):
                    intersection_count += 1
        
        union_count = actual_bins_occupied + fitted_bins_occupied - intersection_count
        
        iou = float(intersection_count / union_count) if union_count > 0 else 0.0
        precision = float(intersection_count / fitted_bins_occupied) if fitted_bins_occupied > 0 else 0.0
        recall = float(intersection_count / actual_bins_occupied) if actual_bins_occupied > 0 else 0.0
        
        metrics = {
            'delta_value': round(delta_value, 4),
            'delta_norm': round(delta_norm, 6),
            'iou': round(iou, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'delta_p95': round(delta_p95, 4),
        }
        
        return metrics
    
    def _poly_equation_string(self, coeffs: np.ndarray) -> str:
        """Generate human-readable polynomial equation string."""
        degree = len(coeffs) - 1
        terms = []
        
        for i, coeff in enumerate(coeffs):
            power = degree - i
            if abs(coeff) < 1e-10:
                continue
            
            term = f"{coeff:.4f}"
            if power == 0:
                terms.append(term)
            elif power == 1:
                terms.append(f"{term}*x")
            else:
                terms.append(f"{term}*x^{power}")
        
        return " + ".join(terms).replace("+ -", "- ")
    
    def generate_digitized_graphs(self, results: Dict[str, Any], instance_dir: str, 
                                   timestamp: str) -> Dict[str, str]:
        """
        Generate output images:
        1. A standalone digitized chart with axis labels, legend and grid.
        2. An overlay image with fitted curves drawn on the original image
           (same pixel dimensions) for visual comparison.
        
        Args:
            results: Processing results dict with curves and fit data
            instance_dir: Per-instance output directory for this processing run
            timestamp: Timestamp string for filenames
            
        Returns:
            Dictionary mapping graph names to file paths
        """
        instance_path = Path(instance_dir)
        instance_path.mkdir(parents=True, exist_ok=True)
        
        saved_graphs = {}
        curves_data = results.get('curves', {})
        axis_info = results.get('axis_info', {})
        
        x_label = f"{axis_info.get('xUnit', 'X')}"
        y_label = f"{axis_info.get('yUnit', 'Y')}"
        
        # Map color names to matplotlib-compatible colors
        color_map = {
            'red': '#e74c3c', 'blue': '#2196F3', 'green': '#27ae60',
            'yellow': '#f1c40f', 'orange': '#e67e22', 'purple': '#9b59b6',
            'black': '#2c3e50', 'gray': '#95a5a6', 'magenta': '#e91e63',
            'pink': '#ff69b4', 'light blue': '#03a9f4', 'cyan': '#00bcd4',
            'light green': '#8bc34a', 'dark blue': '#1a237e', 'dark red': '#b71c1c',
            'brown': '#795548', 'teal': '#009688',
        }
        # Extra cycle for curves keyed by non-standard names (gray_0, etc.)
        _gs_cycle = ['#e74c3c', '#2196F3', '#27ae60', '#e67e22',
                     '#9b59b6', '#00bcd4', '#795548', '#e91e63',
                     '#f1c40f', '#1a237e']
        
        # ═══════════════════════════════════════════════════════
        # 1. Standalone digitized chart (axis coords, legend)
        # ═══════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(10, 7))
        has_valid_curves = False
        
        for color_name, curve_data in curves_data.items():
            fit = curve_data.get('fit_result', {})
            fitted_points = fit.get('fitted_points', [])
            if not fitted_points:
                continue
            
            has_valid_curves = True
            x_vals = [p['x'] for p in fitted_points]
            y_vals = [p['y'] for p in fitted_points]
            label_text = curve_data.get('label', color_name)
            r_sq = fit.get('r_squared', 0)
            plot_color = color_map.get(color_name.lower(), None)
            if plot_color is None:
                ci = list(curves_data.keys()).index(color_name)
                plot_color = _gs_cycle[ci % len(_gs_cycle)]
            
            ax.plot(x_vals, y_vals, color=plot_color, linewidth=2.5,
                    label=f"{label_text} (R\u00b2={r_sq:.4f})")
        
        if has_valid_curves:
            ax.set_xlabel(x_label, fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.legend(loc='best', fontsize=9, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(self.xMin, self.xMax)
            ax.set_ylim(self.yMin, self.yMax)
            fig.tight_layout()
            
            combined_path = str(instance_path / f"digitized_curves_{timestamp}.png")
            fig.savefig(combined_path, dpi=150, bbox_inches='tight')
            saved_graphs['all_curves'] = combined_path
        
        plt.close(fig)
        
        # ═══════════════════════════════════════════════════════
        # 2. Overlay image (fitted curves on original, same dims)
        # ═══════════════════════════════════════════════════════
        image_path = results.get('image_path', '')
        if has_valid_curves and image_path and Path(image_path).exists():
            orig_img = np.array(Image.open(image_path).convert('RGB'))
            img_h, img_w = orig_img.shape[:2]

            pa = results.get('plot_area', {})
            pa_left  = pa.get('left', 0)
            pa_top   = pa.get('top', 0)
            pa_right = pa.get('right', img_w)
            pa_bot   = pa.get('bottom', img_h)

            x_min, x_max = self.xMin, self.xMax
            y_min, y_max = self.yMin, self.yMax
            ax_w = x_max - x_min if x_max != x_min else 1.0
            ax_h = y_max - y_min if y_max != y_min else 1.0

            # Fence-post: match normalize_to_axis which uses (p_right - p_left - 1)
            pa_px_w = max(pa_right - pa_left - 1, 1)
            pa_px_h = max(pa_bot - pa_top - 1, 1)

            def ax2px_x(ax_x):
                return pa_left + (ax_x - x_min) / ax_w * pa_px_w

            def ax2px_y(ax_y):
                return pa_top + (y_max - ax_y) / ax_h * pa_px_h

            dpi = 150
            fig2, ax2 = plt.subplots(figsize=(img_w / dpi, img_h / dpi), dpi=dpi)
            ax2.imshow(orig_img, extent=[0, img_w, img_h, 0], aspect='auto')

            for color_name, curve_data in curves_data.items():
                fit = curve_data.get('fit_result', {})
                fitted_points = fit.get('fitted_points', [])
                if not fitted_points:
                    continue

                px_x = [ax2px_x(p['x']) for p in fitted_points]
                px_y = [ax2px_y(p['y']) for p in fitted_points]

                plot_color = color_map.get(color_name.lower(), None)
                if plot_color is None:
                    ci = list(curves_data.keys()).index(color_name)
                    plot_color = _gs_cycle[ci % len(_gs_cycle)]

                ax2.plot(px_x, px_y, color=plot_color, linewidth=2)

            ax2.set_xlim(0, img_w)
            ax2.set_ylim(img_h, 0)
            ax2.axis('off')
            fig2.subplots_adjust(left=0, right=1, top=1, bottom=0)

            overlay_path = str(instance_path / f"overlay_{timestamp}.png")
            fig2.savefig(overlay_path, dpi=dpi, bbox_inches='tight',
                         pad_inches=0)
            saved_graphs['overlay'] = overlay_path

            plt.close(fig2)
        
        return saved_graphs
    
    def process_curve_image(self, image_path: str, features: List[Dict], 
                           output_dir: str = "./output/",
                           crop_box: Optional[Tuple[int, int, int, int]] = None,
                           mode: str = "auto",
                           *,
                           anchors: Optional[List[Any]] = None,
                           ignore_dashed: bool = True,
                           smoothing_strength: int = 0,
                           use_skeleton_bw: bool = True,
                           target_curves: int = 0,
                           dashed_threshold: float = 0.45,
                           text_threshold: float = 0.50,
                           plot_area_override: Optional[Tuple[int, int, int, int]] = None,
                           exclude_curve_mode: str = "",
                           ) -> Dict[str, Any]:
        """
        Complete end-to-end processing of a curve image.
        
        Args:
            image_path: Path to image file
            features: Feature list from Gemini (curves with colors)
            crop_box: Optional crop coordinates
            mode: 'auto' (detect), 'color', 'grayscale', or 'bw'
            anchors: Optional list of ((start_x, start_y), (end_x, end_y))
                     for anchor-guided B/W tracing
            ignore_dashed: Whether to reject dashed/dotted lines (B/W only)
            smoothing_strength: Savitzky-Golay window for B/W smoothing (0=auto)
            use_skeleton_bw: If True, use new skeleton-based B/W pipeline;
                             if False, use legacy column-scan pipeline.
            target_curves: Override curve count (0 = auto-detect).
            dashed_threshold: Threshold for dashed line rejection.
            text_threshold: Threshold for text component rejection.
            plot_area_override: Manual pixel bounds (left, top, right, bottom).
            
        Returns:
            Dictionary with processed curves and fitted polynomials
        """
        image = self.load_image(image_path)
        image = self.crop_image(image, crop_box)
        
        width, height = image.size

        # --- Mode Router: HSV-based auto-detection ---
        # Map legacy mode names for backward compatibility
        mode_map = {'grayscale': 'bw', 'color': 'color', 'auto': 'auto'}
        router_mode = mode_map.get(mode, mode)
        detected_mode = classify_image_mode(image, mode_override=router_mode)
        grayscale_mode = (detected_mode == 'bw')

        # Detect actual plot area boundaries for accurate coordinate mapping
        if plot_area_override:
            plot_area = plot_area_override
        else:
            plot_area = self.detect_plot_area(image)
            # Tick-based refinement is useful for B/W charts but can bias
            # coloured charts upward (ticks are not always at true bounds).
            if grayscale_mode:
                plot_area = self._refine_plot_area_with_ticks(image, plot_area)
        
        results = {
            'image_path': image_path,
            'image_dimensions': {'width': width, 'height': height},
            'plot_area': {'left': plot_area[0], 'top': plot_area[1], 
                         'right': plot_area[2], 'bottom': plot_area[3]},
            'axis_info': self.axis_info,
            'curves': {}
        }
        
        results['grayscale_mode'] = bool(grayscale_mode)
        results['detected_mode'] = detected_mode
        
        if grayscale_mode and 'curves' in features:
            # ─── Grayscale path ───
            # Filter GPT features: remove surge/reference/axis entries
            import re as _re
            _NON_CURVE_KW = {'surge', 'axis', 'boundary', 'limit',
                             'reference', 'design', 'dashed',
                             'flow line', 'flowline', 'guide line', 'guideline'}
            curve_features = [
                cf for cf in features['curves']
                if not any(kw in cf.get('label', '').lower()
                           for kw in _NON_CURVE_KW)
                and not _re.match(r'^[\d.,\s]+$',
                                  cf.get('label', '').strip())
                and not _re.search(r'\bflow\b.*\bline\b|\bline\b.*\bflow\b',
                                   cf.get('label', '').lower())
            ]
            
            num_curves = len(curve_features)
            # If user explicitly set target_curves in sidebar, prefer that
            if target_curves > 0:
                num_curves = target_curves
            # When anchors are provided, the user explicitly defined
            # the curves — use anchor count as the definitive number.
            _anchor_mode = False
            if anchors:
                _norm_anchors = [a for a in anchors
                                 if isinstance(a, (list, tuple)) and len(a) >= 2]
                if _norm_anchors:
                    num_curves = len(_norm_anchors)
                    _anchor_mode = True
            if num_curves > 0:
                # Choose B/W extraction method
                if use_skeleton_bw:
                    # ── NEW: skeleton-based extraction with dashed rejection,
                    #    anchor tracing, endpoint extension, and smoothing ──
                    gray_clusters = extract_bw_curves(
                        image, num_curves, plot_area,
                        anchors=anchors,
                        ignore_dashed=ignore_dashed,
                        dashed_threshold=dashed_threshold,
                        text_threshold=text_threshold,
                        smoothing_strength=smoothing_strength,
                        extend_ends=True,
                        exclude_curve_mode=exclude_curve_mode,
                    )
                    results['extraction_method'] = 'skeleton'
                    # Stash skeleton for debug saving later
                    _skel, _bin, _skel_pa = get_last_skeleton()
                    results['_debug_skeleton'] = _skel
                    results['_debug_binary'] = _bin
                    results['_debug_skeleton_plot_area'] = _skel_pa
                else:
                    # ── LEGACY: column-scan tracker ──
                    gray_clusters = self.extract_curves_grayscale(
                        image, num_curves, plot_area
                    )
                    results['extraction_method'] = 'column_scan'
                
                # ── Sort labels to match spatial cluster order ──
                # Clusters are sorted top-first (smallest y-pixel).
                # On a standard chart the topmost curve has the
                # highest numeric value, so sort labels descending.
                def _label_num(cf):
                    nums = _re.findall(r'[\d.]+',
                                       str(cf.get('label', '')))
                    return float(nums[0]) if nums else 0.0

                curve_features_sorted = sorted(
                    curve_features,
                    key=_label_num,
                    reverse=(self.yMax >= self.yMin),
                )

                # If extraction found more components than the LLM
                # reported, build extra labels by extrapolating the
                # numeric series the LLM gave us.  Clusters are sorted
                # top-first, i.e. highest chart value first, so any
                # extra curves the LLM missed are most likely at the
                # top (higher values).
                n_clusters = len(gray_clusters)
                if n_clusters > len(curve_features_sorted):
                    known_vals = sorted(
                        [_label_num(cf) for cf in curve_features_sorted],
                        reverse=True,          # descending: highest first
                    )
                    if len(known_vals) >= 2:
                        step = abs(known_vals[0] - known_vals[1])
                        if step == 0:
                            step = 10
                    else:
                        step = 10

                    n_extra = n_clusters - len(known_vals)
                    # Build labels for ALL clusters: existing labels +
                    # generic numbered labels for extras.  Avoids
                    # generating wrong values via extrapolation.
                    new_features = []
                    for i in range(n_clusters):
                        if i < len(curve_features_sorted):
                            new_features.append(dict(curve_features_sorted[i]))
                        else:
                            new_features.append({
                                'color': f'gray_{i}',
                                'label': f'Curve {i + 1}',
                            })
                    curve_features_sorted = new_features

                # ── Debug: save cluster visualisation ──
                if _DEBUG_DIR:
                    _dbg_cluster_img = np.array(image.convert("RGB")).copy()
                    _dbg_colors = [(255,0,0),(0,180,0),(0,0,255),
                                   (255,165,0),(128,0,255),(0,200,200)]
                    for _ci, _pxs in sorted(gray_clusters.items()):
                        _cc = _dbg_colors[_ci % len(_dbg_colors)]
                        for _px, _py in _pxs:
                            if 0 <= _py < _dbg_cluster_img.shape[0] and 0 <= _px < _dbg_cluster_img.shape[1]:
                                _dbg_cluster_img[_py, _px] = _cc
                    _save_debug_image("gs_clusters_overlay", _dbg_cluster_img)

                # ── Fit each grayscale cluster using the SAME logic as colour path ──
                # normalize_to_axis → clean_coordinates_local → fit_polynomial_curve(degree=2)
                # Build stable cluster order for label assignment:
                # use right-end extent (higher-speed curves usually extend further right).
                cluster_items = sorted(gray_clusters.items())
                if cluster_items:
                    def _cluster_right_extent(pixels: List[Tuple[int, int]]) -> float:
                        xs = [p[0] for p in pixels]
                        if not xs:
                            return 0.0
                        return float(np.percentile(np.array(xs, dtype=np.float64), 95))

                    cluster_order = [
                        idx for idx, _ in sorted(
                            cluster_items,
                            key=lambda kv: _cluster_right_extent(kv[1]),
                            reverse=True,
                        )
                    ]
                else:
                    cluster_order = []

                label_by_cluster: Dict[int, str] = {}
                for rank, cluster_idx in enumerate(cluster_order):
                    if rank < len(curve_features_sorted):
                        cf = curve_features_sorted[rank]
                        label_by_cluster[cluster_idx] = cf.get(
                            'label', f'Curve {cluster_idx + 1}'
                        )
                    else:
                        label_by_cluster[cluster_idx] = f'Curve {cluster_idx + 1}'

                for curve_idx, pixels in cluster_items:
                    label = label_by_cluster.get(curve_idx, f'Curve {curve_idx + 1}')
                    color_key = f'gray_{curve_idx}'

                    if len(pixels) < 2:
                        results['curves'][color_key] = {
                            'label': label, 'color': color_key,
                            'extraction_mode': 'grayscale',
                            'error': f'Insufficient pixels ({len(pixels)}) for curve {curve_idx}'
                        }
                        continue

                    # Normalize pixel coords → axis coords (same as colour path)
                    axis_coords = self.normalize_to_axis(pixels, width, height, plot_area)

                    # ── BW-specific robust polynomial fitting ──
                    # Replaces the generic clean→sigma-clip→polyfit(2) pipeline
                    # with binned-median centerline extraction, Hampel + sigma-clip
                    # outlier removal, Savitzky-Golay pre-smooth, and BIC-selected
                    # polynomial degree 1-4 with shape sanity checks.
                    fit_result, cleaned_coords = fit_bw_curve(axis_coords)

                    # Clamp fitted y-values to axis range (prevents polynomial
                    # extrapolation from producing physically impossible values
                    # like negative power or efficiency > 100%).
                    for p in fit_result.get('fitted_points', []):
                        p['y'] = max(self.yMin, min(self.yMax, p['y']))

                    reproj_info = {}

                    # Quality metrics (same as colour path)
                    metrics = self.calculate_curve_metrics(cleaned_coords, fit_result)

                    # Peak / best-value point
                    peak_point = fit_result.get('peak_point')

                    results['curves'][color_key] = {
                        'label': label, 'color': color_key,
                        'extraction_mode': 'grayscale',
                        'raw_pixel_points': pixels,
                        'raw_axis_coords': [
                            {'x': round(x, 4), 'y': round(y, 4)}
                            for x, y in sorted(set(axis_coords))
                        ],
                        'plot_area': list(plot_area),
                        'original_point_count': len(pixels),
                        'normalized_point_count': len(axis_coords),
                        'cleaned_point_count': len(cleaned_coords),
                        'fit_result': fit_result,
                        'metrics': metrics,
                        'peak_point': peak_point,
                        'reprojection': reproj_info,
                    }

                # ── Surge / dashed-line extraction ──
                # Skip in anchor mode: the user explicitly defined
                # which curves to extract — don't add spurious fragments.
                if not _anchor_mode:
                    try:
                        surge_curves = extract_surge_lines(
                            image, plot_area, gray_clusters,
                        )
                        for s_idx, s_pixels in surge_curves.items():
                            s_key = f'surge_{s_idx}'
                            s_axis = self.normalize_to_axis(s_pixels, width, height, plot_area)
                            s_fit, s_cleaned = fit_bw_curve(s_axis)
                            s_peak = s_fit.get('peak_point')
                            results['curves'][s_key] = {
                                'label': f'Surge Line {s_idx + 1}',
                                'color': s_key,
                                'extraction_mode': 'grayscale',
                                'is_surge_line': True,
                                'raw_pixel_points': s_pixels,
                                'plot_area': list(plot_area),
                                'original_point_count': len(s_pixels),
                                'normalized_point_count': len(s_axis),
                                'cleaned_point_count': len(s_cleaned),
                                'fit_result': s_fit,
                                'peak_point': s_peak,
                            }
                    except Exception as _surge_err:
                        logger.warning('Surge line extraction failed: %s', _surge_err)

                # ── Debug: save fitted-curves overlay ──
                if _DEBUG_DIR:
                    try:
                        _fig, _ax = plt.subplots(figsize=(10, 6))
                        _ax.set_xlim(self.xMin, self.xMax)
                        _ax.set_ylim(self.yMin, self.yMax)
                        _fit_colors = ['red', 'green', 'blue',
                                       'orange', 'purple', 'cyan']
                        for _fi, (ck, cv) in enumerate(results['curves'].items()):
                            fr = cv.get('fit_result', {})
                            fp = fr.get('fitted_points', [])
                            if fp:
                                _ax.plot([p['x'] for p in fp],
                                         [p['y'] for p in fp],
                                         color=_fit_colors[_fi % len(_fit_colors)],
                                         linewidth=2,
                                         label=cv.get('label', ck))
                        _ax.legend(fontsize=8)
                        _ax.set_title("Grayscale: fitted curves")
                        _dbg_path = Path(_DEBUG_DIR) / "gs_fitted_curves.png"
                        _fig.savefig(str(_dbg_path), dpi=120)
                        plt.close(_fig)
                    except Exception:
                        pass

                # ──── Self-validation: y-range consistency check ────
                all_fitted_y = []
                for ck, cv in results['curves'].items():
                    fr = cv.get('fit_result', {})
                    fp = fr.get('fitted_points', [])
                    all_fitted_y.extend([p['y'] for p in fp])

                if all_fitted_y:
                    y_data_min = min(all_fitted_y)
                    y_data_max = max(all_fitted_y)
                    y_axis_range = self.yMax - self.yMin
                    y_data_range = y_data_max - y_data_min

                    results['validation'] = {
                        'y_data_min': round(y_data_min, 4),
                        'y_data_max': round(y_data_max, 4),
                        'y_axis_min': self.yMin,
                        'y_axis_max': self.yMax,
                        'y_coverage_pct': round(
                            y_data_range / max(y_axis_range, 1e-9) * 100, 1
                        ),
                        'plot_area_pixels': list(plot_area),
                    }

                    # Warn if data occupies less than 20% of axis range
                    if y_axis_range > 0 and y_data_range < y_axis_range * 0.20:
                        results['validation']['warning'] = (
                            f"Data y-range ({y_data_min:.1f}–{y_data_max:.1f}) "
                            f"covers only {y_data_range/y_axis_range*100:.0f}% "
                            f"of axis range ({self.yMin}–{self.yMax}). "
                            "Plot area detection may be inaccurate; "
                            "consider using Plot Area Override."
                        )
        
        elif 'curves' in features:
            # ─── Colour path: dynamic RGB extraction ───
            _DARK_NAMES = {'black', 'gray', 'grey', 'dark gray', 'dark grey'}
            all_colors = [cf.get('color', 'unknown').lower()
                          for cf in features['curves']]
            has_colored_series = any(c not in _DARK_NAMES for c in all_colors)

            # Build anchor lookup: anchor_pairs[i] → curve index i
            _anchor_list = anchors or []
            _color_curve_idx = -1   # running index of non-skipped curves

            for idx, curve_feature in enumerate(features['curves']):
                color = curve_feature.get('color', 'unknown')
                label = curve_feature.get('label', f'Curve {idx+1}')
                
                # Skip dark / axis colours when real coloured curves exist
                if color.lower() in _DARK_NAMES and has_colored_series:
                    continue
                
                # Skip surge line — it's a reference boundary, not a performance curve
                if 'surge' in label.lower():
                    continue

                _color_curve_idx += 1
                
                # Extract pixels using dynamic RGB range adaptation
                pixels = self.extract_color_pixels_dynamic(image, color)
                
                # ── Filter pixels to within the plot area ──
                p_left, p_top, p_right, p_bottom = plot_area
                pixels = [(x, y) for x, y in pixels
                          if p_left <= x <= p_right and p_top <= y <= p_bottom]
                
                # Spatial continuity filter: if an anchor pair exists for
                # this curve, prefer the blob nearest to the anchors;
                # otherwise keep the largest connected blob.
                if _color_curve_idx < len(_anchor_list) and _anchor_list[_color_curve_idx]:
                    a_start, a_end = _anchor_list[_color_curve_idx]
                    pixels = self.filter_spatially_near_anchor(
                        pixels, width, height, a_start, a_end)
                else:
                    pixels = self.filter_spatially_connected(pixels, width, height)
                
                if len(pixels) < 2:
                    results['curves'][color] = {
                        'label': label,
                        'color': color,
                        'error': f'Insufficient pixels extracted for color {color}'
                    }
                    continue
                
                # Normalize to axis coordinates using detected plot area
                axis_coords = self.normalize_to_axis(pixels, width, height, plot_area)
                
                # Clean with shape-preserving local filter
                cleaned_coords = self.clean_coordinates_local(axis_coords)
                
                # Polynomial fit (degree 2 — performance curves are parabolic)
                fit_result = self.fit_polynomial_curve(cleaned_coords, degree=2)
                
                # Calculate quality metrics
                metrics = self.calculate_curve_metrics(cleaned_coords, fit_result)
                
                # Peak / best-value point for colour curves
                color_peak = None
                color_fps = fit_result.get('fitted_points', [])
                if color_fps:
                    cpk = max(color_fps, key=lambda p: p['y'])
                    color_peak = {'x': round(cpk['x'], 4), 'y': round(cpk['y'], 4)}

                results['curves'][color] = {
                    'label': label,
                    'color': color,
                    'plot_area': list(plot_area),
                    'original_point_count': len(pixels),
                    'normalized_point_count': len(axis_coords),
                    'cleaned_point_count': len(cleaned_coords),
                    'fit_result': fit_result,
                    'metrics': metrics,
                    'peak_point': color_peak,
                }
        
        # Compute aggregate (graph-level) metrics across all curves
        all_metrics = [c.get('metrics', {}) for c in results['curves'].values()
                       if c.get('metrics', {}).get('delta_value') is not None]
        if all_metrics:
            results['overall_metrics'] = {
                'delta_value': round(float(np.mean([m['delta_value'] for m in all_metrics])), 4),
                'delta_norm': round(float(np.mean([m['delta_norm'] for m in all_metrics])), 6),
                'iou': round(float(np.mean([m['iou'] for m in all_metrics])), 4),
                'precision': round(float(np.mean([m['precision'] for m in all_metrics])), 4),
                'recall': round(float(np.mean([m['recall'] for m in all_metrics])), 4),
                'delta_p95': round(float(np.max([m['delta_p95'] for m in all_metrics])), 4),
                'curve_count': len(all_metrics),
            }
        else:
            results['overall_metrics'] = {}
        
        # Create per-instance output folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        instance_dir = str(Path(output_dir) / timestamp)
        Path(instance_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate digitized output graph
        saved_graphs = self.generate_digitized_graphs(results, instance_dir, timestamp)
        results['output_graphs'] = saved_graphs
        results['instance_dir'] = instance_dir

        # Save skeleton debug image to output folder
        _dbg_skel = results.pop('_debug_skeleton', None)
        _dbg_bin = results.pop('_debug_binary', None)
        _dbg_skel_pa = results.pop('_debug_skeleton_plot_area', None)
        if _dbg_skel is not None:
            try:
                skel_img = (_dbg_skel.astype(np.uint8) * 255)
                skel_path = str(Path(instance_dir) / f"skeleton_{timestamp}.png")
                Image.fromarray(skel_img).save(skel_path)
                saved_graphs['skeleton'] = skel_path
                logger.info("Saved skeleton debug image: %s", skel_path)
            except Exception as _skel_err:
                logger.debug("Failed to save skeleton debug image: %s", _skel_err)
        if _dbg_bin is not None:
            try:
                bin_img = (_dbg_bin.astype(np.uint8) * 255)
                bin_path = str(Path(instance_dir) / f"binary_{timestamp}.png")
                Image.fromarray(bin_img).save(bin_path)
                saved_graphs['binary'] = bin_path
                logger.info("Saved binary debug image: %s", bin_path)
            except Exception as _bin_err:
                logger.debug("Failed to save binary debug image: %s", _bin_err)

        return results
