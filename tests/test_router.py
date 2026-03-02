"""
Unit tests for core.router – image-mode classification.
"""

import numpy as np
import pytest
from PIL import Image

from core.router import classify_image_mode


def _make_color_image(w=200, h=150):
    """Create a synthetic color image with high saturation."""
    arr = np.ones((h, w, 3), dtype=np.uint8) * 200  # light gray bg (below 240)
    # Red band across the middle (saturated, V in range)
    arr[30:70, :, 0] = 220
    arr[30:70, :, 1] = 40
    arr[30:70, :, 2] = 40
    # Blue band
    arr[70:110, :, 0] = 40
    arr[70:110, :, 1] = 40
    arr[70:110, :, 2] = 220
    # Green stripe
    arr[110:130, :, 0] = 40
    arr[110:130, :, 1] = 200
    arr[110:130, :, 2] = 40
    return Image.fromarray(arr)


def _make_bw_image(w=200, h=150):
    """Create a synthetic B/W image (grayscale-like, no saturation)."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, :] = 240  # white background
    # Black curve
    arr[60:65, 20:180, :] = 10
    # Gray gridlines
    for x in range(0, w, 40):
        arr[:, x:x + 1, :] = 180
    return Image.fromarray(arr)


class TestClassifyImageMode:

    def test_auto_color_detected(self):
        img = _make_color_image()
        result = classify_image_mode(img, mode_override="auto")
        assert result == "color"

    def test_auto_bw_detected(self):
        img = _make_bw_image()
        result = classify_image_mode(img, mode_override="auto")
        assert result == "bw"

    def test_force_color(self):
        """Forcing color mode on a B/W image should return 'color'."""
        img = _make_bw_image()
        result = classify_image_mode(img, mode_override="color")
        assert result == "color"

    def test_force_bw(self):
        """Forcing bw mode on a color image should return 'bw'."""
        img = _make_color_image()
        result = classify_image_mode(img, mode_override="bw")
        assert result == "bw"

    def test_grayscale_input_image(self):
        """A pure grayscale (mode 'L') image should be classified as bw."""
        arr = np.random.randint(100, 250, (100, 100), dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        result = classify_image_mode(img, mode_override="auto")
        assert result == "bw"

    def test_rgba_color_image(self):
        """RGBA color image should still be detected as color."""
        img = _make_color_image().convert("RGBA")
        result = classify_image_mode(img, mode_override="auto")
        assert result == "color"
