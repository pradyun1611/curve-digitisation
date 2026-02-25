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
        6. Connected-component labelling (8-connectivity).
        7. Filter components by horizontal extent (real curves span the plot).
        8. Sort by mean y (topmost = index 0).
        9. Column-median thinning for a 1-px skeleton per curve.

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
        binary = gray < threshold
        _save_debug_image("gs_binary", binary)

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
        _save_debug_image("gs_after_grid_suppress", binary)

        # Small opening to remove residual specks after grid removal
        open_kernel = np.ones((2, 2), dtype=bool)
        binary = binary_opening(binary, structure=open_kernel, iterations=1)

        # ── Step 5: Connected components (8-connectivity) ──
        structure_8 = np.ones((3, 3), dtype=int)
        labelled, n_components = ndimage_label(binary, structure=structure_8)
        logger.debug("grayscale extraction: %d components before filtering", n_components)

        # ── Step 6: Filter by horizontal extent ──
        min_width = max(10, int(region_w * 0.10))   # curve must span ≥ 10% of plot

        valid_components = []
        for comp_id in range(1, n_components + 1):
            ys, xs = np.where(labelled == comp_id)
            n_px = len(ys)
            if n_px < 5:
                continue
            h_extent = int(xs.max()) - int(xs.min()) + 1
            v_extent = int(ys.max()) - int(ys.min()) + 1

            if h_extent < min_width:
                continue

            aspect = h_extent / max(v_extent, 1)

            # Skip full-width very thin horizontal lines (reference/axis remnants)
            if h_extent > region_w * 0.85 and aspect > 40:
                continue
            # Skip full-height very narrow vertical lines
            if v_extent > region_h * 0.85 and aspect < 0.03:
                continue

            mean_y = float(np.mean(ys))
            valid_components.append((comp_id, mean_y, n_px))

        # ── Step 7: Sort by mean y-position (top-of-image first) ──
        valid_components.sort(key=lambda x: x[1])
        logger.debug("grayscale extraction: %d valid curves after filtering", len(valid_components))

        # ── Step 8: Column-median thinning per component ──
        result: Dict[int, List[Tuple[int, int]]] = {}
        for idx, (comp_id, _, _) in enumerate(valid_components):
            ys, xs = np.where(labelled == comp_id)

            col_buckets: Dict[int, List[int]] = {}
            for x_val, y_val in zip(xs, ys):
                col_buckets.setdefault(int(x_val), []).append(int(y_val))

            thinned = []
            for cx in sorted(col_buckets):
                median_y = int(np.median(col_buckets[cx]))
                thinned.append((cx + p_left, median_y + p_top))

            result[idx] = thinned

        _save_debug_image("gs_labelled", (labelled > 0).astype(np.uint8) * 255)
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
            p_width = p_right - p_left
            p_height = p_bottom - p_top
        else:
            p_left, p_top = 0, 0
            p_width = image_width
            p_height = image_height
        
        for px, py in pixel_coords:
            # Normalize pixel relative to plot area to 0-1 range
            norm_x = (px - p_left) / p_width if p_width > 0 else 0
            norm_y = 1 - ((py - p_top) / p_height) if p_height > 0 else 0  # Flip Y
            
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
        Generate a combined digitized output graph from fitted polynomial curves.
        
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
        description = axis_info.get('imageDescription', 'Performance Curve')
        
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
        
        # ── Combined plot with all curves ──
        fig, ax = plt.subplots(figsize=(10, 7))
        has_valid_curves = False
        
        for color_name, curve_data in curves_data.items():
            fit = curve_data.get('fit_result', {})
            fitted_points = fit.get('fitted_points', [])
            if not fitted_points:
                continue
            
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
                    label=f"{label_text} (R²={r_sq:.4f})")
        
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
        
        return saved_graphs
    
    def process_curve_image(self, image_path: str, features: List[Dict], 
                           output_dir: str = "./output/",
                           crop_box: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Complete end-to-end processing of a curve image.
        
        Args:
            image_path: Path to image file
            features: Feature list from Gemini (curves with colors)
            crop_box: Optional crop coordinates
            
        Returns:
            Dictionary with processed curves and fitted polynomials
        """
        image = self.load_image(image_path)
        image = self.crop_image(image, crop_box)
        
        width, height = image.size
        
        # Detect actual plot area boundaries for accurate coordinate mapping
        plot_area = self.detect_plot_area(image)
        
        results = {
            'image_path': image_path,
            'image_dimensions': {'width': width, 'height': height},
            'plot_area': {'left': plot_area[0], 'top': plot_area[1], 
                         'right': plot_area[2], 'bottom': plot_area[3]},
            'axis_info': self.axis_info,
            'curves': {}
        }
        
        # Auto-detect grayscale vs colour image
        grayscale_mode = self.is_grayscale_image(image)
        results['grayscale_mode'] = bool(grayscale_mode)
        
        if grayscale_mode and 'curves' in features:
            # ─── Grayscale path: connected-component extraction ───
            # Filter GPT features: remove surge/reference/axis entries
            import re as _re
            _NON_CURVE_KW = {'surge', 'axis', 'boundary', 'limit',
                             'reference', 'design', 'dashed'}
            curve_features = [
                cf for cf in features['curves']
                if not any(kw in cf.get('label', '').lower()
                           for kw in _NON_CURVE_KW)
                and not _re.match(r'^[\d.,\s]+$',
                                  cf.get('label', '').strip())
            ]
            
            num_curves = len(curve_features)
            if num_curves > 0:
                gray_clusters = self.extract_curves_grayscale(
                    image, num_curves, plot_area
                )
                
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
                        step = known_vals[0] - known_vals[1]    # gap between adjacent
                        if step == 0:
                            step = 10
                    else:
                        step = 10

                    n_extra = n_clusters - len(known_vals)
                    # Extrapolate *upward* from the highest known value
                    top_val = known_vals[0] + step * n_extra
                    full_vals = [top_val - i * step
                                 for i in range(n_clusters)]

                    # Re-build curve_features_sorted with correct count
                    new_features = []
                    for i, val in enumerate(full_vals):
                        lbl = str(int(val)) if val == int(val) else str(val)
                        if i < n_extra:
                            # brand-new extrapolated entry (above LLM range)
                            cf_copy = {'color': f'gray_{i}', 'label': lbl}
                        else:
                            # reuse existing LLM feature, override label
                            cf_copy = dict(
                                curve_features_sorted[i - n_extra])
                            cf_copy['label'] = lbl
                        new_features.append(cf_copy)
                    curve_features_sorted = new_features

                for cluster_idx, pixels in gray_clusters.items():
                    if cluster_idx < len(curve_features_sorted):
                        cf = curve_features_sorted[cluster_idx]
                        color_key = cf.get('color', f'gray_{cluster_idx}')
                        label = cf.get('label', f'Curve {cluster_idx + 1}')
                    else:
                        color_key = f'gray_{cluster_idx}'
                        label = f'Curve {cluster_idx + 1}'
                    
                    if len(pixels) < 2:
                        results['curves'][color_key] = {
                            'label': label, 'color': color_key,
                            'error': f'Insufficient pixels in component {cluster_idx}'
                        }
                        continue
                    
                    axis_coords = self.normalize_to_axis(pixels, width, height, plot_area)
                    cleaned_coords = self.clean_coordinates_local(axis_coords)
                    # Polynomial fit for grayscale: performance curves are
                    # inherently parabolic, so degree-2 gives a clean shape
                    # without the subtle oscillations a spline can introduce.
                    fit_result = self.fit_polynomial_curve(cleaned_coords, degree=2)
                    metrics = self.calculate_curve_metrics(cleaned_coords, fit_result)
                    
                    results['curves'][color_key] = {
                        'label': label, 'color': color_key,
                        'extraction_mode': 'grayscale',
                        'raw_pixel_points': pixels,
                        'plot_area': list(plot_area),
                        'original_point_count': len(pixels),
                        'normalized_point_count': len(axis_coords),
                        'cleaned_point_count': len(cleaned_coords),
                        'fit_result': fit_result, 'metrics': metrics
                    }
        
        elif 'curves' in features:
            # ─── Colour path: dynamic RGB extraction ───
            _DARK_NAMES = {'black', 'gray', 'grey', 'dark gray', 'dark grey'}
            all_colors = [cf.get('color', 'unknown').lower()
                          for cf in features['curves']]
            has_colored_series = any(c not in _DARK_NAMES for c in all_colors)

            for idx, curve_feature in enumerate(features['curves']):
                color = curve_feature.get('color', 'unknown')
                label = curve_feature.get('label', f'Curve {idx+1}')
                
                # Skip dark / axis colours when real coloured curves exist
                if color.lower() in _DARK_NAMES and has_colored_series:
                    continue
                
                # Skip surge line — it's a reference boundary, not a performance curve
                if 'surge' in label.lower():
                    continue
                
                # Extract pixels using dynamic RGB range adaptation
                pixels = self.extract_color_pixels_dynamic(image, color)
                
                # ── Filter pixels to within the plot area ──
                p_left, p_top, p_right, p_bottom = plot_area
                pixels = [(x, y) for x, y in pixels
                          if p_left <= x <= p_right and p_top <= y <= p_bottom]
                
                # Spatial continuity filter: keep only the largest connected
                # blob to remove stray pixels from adjacent-colour curves
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
                
                results['curves'][color] = {
                    'label': label,
                    'color': color,
                    'plot_area': list(plot_area),
                    'original_point_count': len(pixels),
                    'normalized_point_count': len(axis_coords),
                    'cleaned_point_count': len(cleaned_coords),
                    'fit_result': fit_result,
                    'metrics': metrics
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
        
        return results
