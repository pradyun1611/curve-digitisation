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

import numpy as np
from PIL import Image
import json
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/headless use
import matplotlib.pyplot as plt
from datetime import datetime


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
        
        # Define color ranges (RGB thresholds)
        color_ranges = {
            'red': {'r_min': 200, 'r_max': 255, 'g_max': 100, 'b_max': 100},
            'blue': {'b_min': 150, 'b_max': 255, 'r_max': 50, 'g_max': 80},
            'green': {'g_min': 200, 'g_max': 255, 'r_max': 100, 'b_max': 100},
            'yellow': {'r_min': 200, 'g_min': 200, 'b_max': 100},
            'orange': {'r_min': 200, 'g_min': 100, 'g_max': 200, 'b_max': 50},
            'purple': {'r_min': 150, 'b_min': 150, 'g_max': 100},
            'gray': {'r_min': 100, 'r_max': 200, 'g_min': 100, 'g_max': 200, 'b_min': 100, 'b_max': 200},
            'magenta': {'r_min': 180, 'r_max': 255, 'g_max': 100, 'b_min': 180, 'b_max': 255},
            'pink': {'r_min': 200, 'r_max': 255, 'g_max': 150, 'b_min': 150, 'b_max': 255},
            'light blue': {'r_min': 130, 'r_max': 180, 'g_min': 130, 'g_max': 180, 'b_min': 235, 'b_max': 255},
            'cyan': {'r_max': 100, 'g_min': 180, 'g_max': 255, 'b_min': 200, 'b_max': 255},
            'light green': {'r_min': 100, 'r_max': 200, 'g_min': 200, 'g_max': 255, 'b_max': 150},
            'dark blue': {'r_max': 50, 'g_max': 50, 'b_min': 150, 'b_max': 255},
            'dark red': {'r_min': 139, 'r_max': 200, 'g_max': 50, 'b_max': 50},
            'brown': {'r_min': 120, 'r_max': 200, 'g_min': 50, 'g_max': 120, 'b_max': 80},
            'teal': {'r_max': 80, 'g_min': 128, 'g_max': 200, 'b_min': 128, 'b_max': 200},
        }
        
        target_color = target_color_name.lower()
        
        if target_color not in color_ranges:
            # Fallback: highlight any non-white, non-background pixels
            return self._extract_non_white_pixels(img_array)
        
        color_range = color_ranges[target_color]
        
        # Extract pixels matching color range
        pixels = []
        height, width = img_array.shape[:2]
        
        for y in range(height):
            for x in range(width):
                r, g, b = img_array[y, x, :3]
                
                if self._pixel_matches_color(r, g, b, color_range):
                    pixels.append((x, y))
        
        return pixels
    
    def _pixel_matches_color(self, r: int, g: int, b: int, color_range: Dict) -> bool:
        """Check if RGB values match color range."""
        r_match = (color_range.get('r_min', 0) <= r <= color_range.get('r_max', 255))
        g_match = (color_range.get('g_min', 0) <= g <= color_range.get('g_max', 255))
        b_match = (color_range.get('b_min', 0) <= b <= color_range.get('b_max', 255))
        
        return r_match and g_match and b_match
    
    def _calculate_dynamic_color_range(self, rgb_samples: List[Tuple[int, int, int]], 
                                       margin_std: float = 1.0) -> Dict[str, int]:
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
        
        # Create ranges with margin
        margin_r = max(r_std * margin_std, 5)  # Minimum margin of 5
        margin_g = max(g_std * margin_std, 5)
        margin_b = max(b_std * margin_std, 5)
        
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
        
        # Hardcoded color ranges for initial pass
        color_ranges = {
            'red': {'r_min': 200, 'r_max': 255, 'g_max': 100, 'b_max': 100},
            'blue': {'b_min': 150, 'b_max': 255, 'r_max': 50, 'g_max': 80},
            'green': {'g_min': 200, 'g_max': 255, 'r_max': 100, 'b_max': 100},
            'yellow': {'r_min': 200, 'g_min': 200, 'b_max': 100},
            'orange': {'r_min': 200, 'g_min': 100, 'g_max': 200, 'b_max': 50},
            'purple': {'r_min': 150, 'b_min': 150, 'g_max': 100},
            'gray': {'r_min': 100, 'r_max': 200, 'g_min': 100, 'g_max': 200, 'b_min': 100, 'b_max': 200},
            'magenta': {'r_min': 180, 'r_max': 255, 'g_max': 100, 'b_min': 180, 'b_max': 255},
            'pink': {'r_min': 200, 'r_max': 255, 'g_max': 150, 'b_min': 150, 'b_max': 255},
            'light blue': {'r_min': 130, 'r_max': 180, 'g_min': 130, 'g_max': 180, 'b_min': 235, 'b_max': 255},
            'cyan': {'r_max': 100, 'g_min': 180, 'g_max': 255, 'b_min': 200, 'b_max': 255},
            'light green': {'r_min': 100, 'r_max': 200, 'g_min': 200, 'g_max': 255, 'b_max': 150},
            'dark blue': {'r_max': 50, 'g_max': 50, 'b_min': 150, 'b_max': 255},
            'dark red': {'r_min': 139, 'r_max': 200, 'g_max': 50, 'b_max': 50},
            'brown': {'r_min': 120, 'r_max': 200, 'g_min': 50, 'g_max': 120, 'b_max': 80},
            'teal': {'r_max': 80, 'g_min': 128, 'g_max': 200, 'b_min': 128, 'b_max': 200},
        }
        
        if target_color not in color_ranges:
            # Unknown color: fallback to non-white extraction
            return self._extract_non_white_pixels(img_array)
        
        # ─── PASS 1: Initial extraction with hardcoded range ───
        initial_range = color_ranges[target_color]
        initial_pixels = []
        height, width = img_array.shape[:2]
        
        for y in range(height):
            for x in range(width):
                r, g, b = img_array[y, x, :3]
                if self._pixel_matches_color(r, g, b, initial_range):
                    initial_pixels.append((x, y))
        
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
        
        # ─── PASS 2: Final extraction with dynamic range ───
        final_pixels = []
        for y in range(height):
            for x in range(width):
                r, g, b = img_array[y, x, :3]
                if self._pixel_matches_color(r, g, b, dynamic_range):
                    final_pixels.append((x, y))
        
        return final_pixels if final_pixels else initial_pixels
    
    def _extract_non_white_pixels(self, img_array: np.ndarray) -> List[Tuple[int, int]]:
        """Extract non-white, non-background pixels."""
        pixels = []
        height, width = img_array.shape[:2]
        
        for y in range(height):
            for x in range(width):
                r, g, b = img_array[y, x, :3]
                # Exclude white, light gray, and very light pixels
                brightness = (r + g + b) / 3
                if brightness < 240:
                    pixels.append((x, y))
        
        return pixels
    
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
            
            # Calculate fitted points
            x_min, x_max = X.min(), X.max()
            x_fit = np.linspace(x_min, x_max, 50)
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
        
        # ── Combined plot with all curves ──
        fig, ax = plt.subplots(figsize=(10, 7))
        has_valid_curves = False
        
        for color_name, curve_data in curves_data.items():
            fit = curve_data.get('fit_result', {})
            coeffs = fit.get('coefficients')
            if coeffs is None:
                continue
            
            fitted_points = fit.get('fitted_points', [])
            if not fitted_points:
                continue
            
            has_valid_curves = True
            x_vals = [p['x'] for p in fitted_points]
            y_vals = [p['y'] for p in fitted_points]
            label_text = curve_data.get('label', color_name)
            r_sq = fit.get('r_squared', 0)
            plot_color = color_map.get(color_name.lower(), '#333333')
            
            ax.plot(x_vals, y_vals, color=plot_color, linewidth=2.5,
                    label=f"{label_text} (R²={r_sq:.4f})")
        
        if has_valid_curves:
            ax.set_xlabel(x_label, fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.set_title(f"Digitized Curves — {description}", fontsize=14, fontweight='bold')
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
        
        # Extract unique colors from features
        if 'curves' in features:
            for idx, curve_feature in enumerate(features['curves']):
                color = curve_feature.get('color', 'unknown')
                label = curve_feature.get('label', f'Curve {idx+1}')
                
                # Skip black — typically used for reference/axis lines, not data curves
                if color.lower() == 'black':
                    continue
                
                # Extract pixels using dynamic RGB range adaptation
                pixels = self.extract_color_pixels_dynamic(image, color)
                
                if len(pixels) < 2:
                    results['curves'][color] = {
                        'label': label,
                        'color': color,
                        'error': f'Insufficient pixels extracted for color {color}'
                    }
                    continue
                
                # Normalize to axis coordinates using detected plot area
                axis_coords = self.normalize_to_axis(pixels, width, height, plot_area)
                
                # Clean coordinates with RANSAC
                cleaned_coords = self.clean_coordinates_ransac(axis_coords, threshold=0.05)
                
                # Fit polynomial curve
                fit_result = self.fit_polynomial_curve(cleaned_coords, degree=2)
                
                results['curves'][color] = {
                    'label': label,
                    'color': color,
                    'original_point_count': len(pixels),
                    'normalized_point_count': len(axis_coords),
                    'cleaned_point_count': len(cleaned_coords),
                    'fit_result': fit_result
                }
        
        # Create per-instance output folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        instance_dir = str(Path(output_dir) / timestamp)
        Path(instance_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate digitized output graph
        saved_graphs = self.generate_digitized_graphs(results, instance_dir, timestamp)
        results['output_graphs'] = saved_graphs
        results['instance_dir'] = instance_dir
        
        return results
