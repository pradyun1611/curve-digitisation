"""
Tests for color curve digitization.

Uses tests/data/input_color.png (synthetic image with 5 colored curves).
Verifies that extraction is deterministic and finds all curves.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Ensure project root is importable
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.image_processor import CurveDigitizer

DATA_DIR = _ROOT / "tests" / "data"
ARTIFACTS_DIR = _ROOT / "tests" / "artifacts"
COLOR_IMG = DATA_DIR / "input_color.png"

AXIS_INFO = {
    "xMin": 0,
    "xMax": 100,
    "yMin": 0,
    "yMax": 100,
    "xUnit": "%",
    "yUnit": "%",
}


@pytest.fixture(autouse=True)
def _ensure_test_image():
    """Generate test images if they don't exist."""
    if not COLOR_IMG.exists():
        from scripts.generate_test_images import make_color_image
        make_color_image()


@pytest.fixture()
def digitizer():
    return CurveDigitizer(AXIS_INFO)


@pytest.fixture()
def color_image():
    return Image.open(str(COLOR_IMG)).convert("RGB")


class TestColorExtraction:
    """Tests for colored-curve pixel extraction."""

    def test_image_not_grayscale(self, digitizer, color_image):
        assert not digitizer.is_grayscale_image(color_image)

    @pytest.mark.parametrize("color", ["red", "blue", "green", "orange", "purple"])
    def test_extract_color_finds_pixels(self, digitizer, color_image, color):
        pixels = digitizer.extract_color_pixels(color_image, color)
        assert len(pixels) >= 10, f"Expected ≥10 pixels for {color}, got {len(pixels)}"

    @pytest.mark.parametrize("color", ["red", "blue", "green", "orange", "purple"])
    def test_dynamic_extraction(self, digitizer, color_image, color):
        pixels = digitizer.extract_color_pixels_dynamic(color_image, color)
        assert len(pixels) >= 10, f"Expected ≥10 pixels for dynamic {color}, got {len(pixels)}"

    def test_at_least_five_colors_detected(self, digitizer, color_image):
        """At least 5 distinct colors should each yield >= 10 pixels."""
        colors = ["red", "blue", "green", "orange", "purple"]
        detected = sum(
            1 for c in colors
            if len(digitizer.extract_color_pixels(color_image, c)) >= 10
        )
        assert detected >= 5, f"Only {detected}/5 colors detected"

    def test_deterministic_extraction(self, digitizer, color_image):
        """Two runs should produce identical results (no randomness)."""
        px1 = digitizer.extract_color_pixels(color_image, "red")
        px2 = digitizer.extract_color_pixels(color_image, "red")
        assert px1 == px2


class TestColorPipeline:
    """End-to-end color pipeline test."""

    def test_process_curve_image_color(self, digitizer, tmp_path):
        features = {
            "curves": [
                {"color": "red", "shape": "curved", "label": "Red curve"},
                {"color": "blue", "shape": "curved", "label": "Blue curve"},
                {"color": "green", "shape": "straight", "label": "Green curve"},
                {"color": "orange", "shape": "straight", "label": "Orange curve"},
                {"color": "purple", "shape": "curved", "label": "Purple curve"},
            ]
        }
        results = digitizer.process_curve_image(
            str(COLOR_IMG), features, str(tmp_path)
        )

        curves = results.get("curves", {})
        assert len(curves) >= 5, f"Expected ≥5 curves, got {len(curves)}"

        for name, cdata in curves.items():
            assert "error" not in cdata or cdata["error"] is None, (
                f"Curve {name} has error: {cdata.get('error')}"
            )
            assert cdata.get("original_point_count", 0) > 0

    def test_save_debug_artifacts_on_failure(self, digitizer, color_image):
        """Verify debug mask helper works (for artifact saving in CI)."""
        pixels = digitizer.extract_color_pixels(color_image, "red")
        img_arr = np.array(color_image)
        debug_mask = digitizer._pixels_to_debug_mask(pixels, img_arr.shape)
        assert debug_mask.shape[:2] == img_arr.shape[:2]
        assert debug_mask.max() > 0
