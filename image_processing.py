"""
Image Processing Module

Handles image digitization and curve fitting:
- Image preprocessing and cropping
- Curve color detection and extraction
- Pixel-to-coordinate normalization
- RANSAC noise removal
- Polynomial curve fitting
"""

import numpy as np
from PIL import Image
import json
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures


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
        
        # Define color ranges (HSV-like approximation in RGB)
        color_ranges = {
            'red': {'r_min': 200, 'r_max': 255, 'g_max': 100, 'b_max': 100},
            'blue': {'b_min': 200, 'b_max': 255, 'r_max': 100, 'g_max': 150},
            'green': {'g_min': 200, 'g_max': 255, 'r_max': 100, 'b_max': 100},
            'yellow': {'r_min': 200, 'g_min': 200, 'b_max': 100},
            'orange': {'r_min': 200, 'g_min': 100, 'g_max': 200, 'b_max': 50},
            'purple': {'r_min': 150, 'b_min': 150, 'g_max': 100},
            'black': {'r_max': 50, 'g_max': 50, 'b_max': 50},
            'gray': {'r_min': 100, 'r_max': 200, 'g_min': 100, 'g_max': 200, 'b_min': 100, 'b_max': 200},
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
    
    def normalize_to_axis(self, pixel_coords: List[Tuple[int, int]], 
                         image_width: int, image_height: int) -> List[Tuple[float, float]]:
        """
        Normalize pixel coordinates to axis coordinates.
        
        Args:
            pixel_coords: List of (pixel_x, pixel_y) tuples
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            List of (axis_x, axis_y) tuples normalized to axis bounds
        """
        normalized = []
        
        # Assume image coordinates: (0,0) at top-left, x increases right, y increases down
        # Axis coordinates: typically x increases right, y increases up
        # Need to flip y-axis
        
        for px, py in pixel_coords:
            # Normalize pixel to 0-1 range
            norm_x = px / image_width if image_width > 0 else 0
            norm_y = 1 - (py / image_height) if image_height > 0 else 0  # Flip Y
            
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
    
    def process_curve_image(self, image_path: str, features: List[Dict], 
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
        
        results = {
            'image_path': image_path,
            'image_dimensions': {'width': width, 'height': height},
            'axis_info': self.axis_info,
            'curves': {}
        }
        
        # Extract unique colors from features
        if 'curves' in features:
            for idx, curve_feature in enumerate(features['curves']):
                color = curve_feature.get('color', 'unknown')
                label = curve_feature.get('label', f'Curve {idx+1}')
                
                # Extract pixels
                pixels = self.extract_color_pixels(image, color)
                
                if len(pixels) < 2:
                    results['curves'][color] = {
                        'label': label,
                        'color': color,
                        'error': f'Insufficient pixels extracted for color {color}'
                    }
                    continue
                
                # Normalize to axis coordinates
                axis_coords = self.normalize_to_axis(pixels, width, height)
                
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
        
        return results
