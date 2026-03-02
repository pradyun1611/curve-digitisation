"""
Unit tests for core.calibration – axis calibration + mapping fix.
"""

import numpy as np
import pytest

from core.calibration import (
    CalibrationResult,
    calibrate_simple,
    calibrate_manual,
    pixel_to_data,
    data_to_pixel,
    validate_calibration,
    build_mapping_from_calibration,
    calibrate_from_axis_info,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def simple_axis_info():
    return {
        "xMin": 0, "xMax": 100,
        "yMin": 0, "yMax": 50,
        "xUnit": "Hz", "yUnit": "dB",
    }


@pytest.fixture
def plot_area():
    """(left, top, right, bottom) in pixels."""
    return (50, 30, 450, 330)  # 400 wide, 300 tall


# ---------------------------------------------------------------------------
# calibrate_simple
# ---------------------------------------------------------------------------
class TestCalibrateSimple:

    def test_returns_calibration_result(self, simple_axis_info, plot_area):
        cal = calibrate_simple(simple_axis_info, plot_area)
        assert isinstance(cal, CalibrationResult)
        assert cal.method == "simple"
        assert cal.pixel_to_data is not None
        assert cal.data_to_pixel is not None

    def test_corners_map_correctly(self, simple_axis_info, plot_area):
        """Top-left → (xMin, yMax), bottom-right → (xMax, yMin)."""
        cal = calibrate_simple(simple_axis_info, plot_area)
        left, top, right, bottom = plot_area

        # Top-left of plot area → (0, 50)
        pts = pixel_to_data([(left, top)], cal)
        dx, dy = pts[0]
        assert abs(dx - 0.0) < 0.01
        assert abs(dy - 50.0) < 0.01

        # Bottom-right of plot area → (100, 0)
        pts = pixel_to_data([(right, bottom)], cal)
        dx, dy = pts[0]
        assert abs(dx - 100.0) < 0.01
        assert abs(dy - 0.0) < 0.01

    def test_center_maps_correctly(self, simple_axis_info, plot_area):
        cal = calibrate_simple(simple_axis_info, plot_area)
        left, top, right, bottom = plot_area
        cx, cy = (left + right) / 2, (top + bottom) / 2

        pts = pixel_to_data([(cx, cy)], cal)
        dx, dy = pts[0]
        assert abs(dx - 50.0) < 0.01
        assert abs(dy - 25.0) < 0.01

    def test_round_trip(self, simple_axis_info, plot_area):
        """pixel→data→pixel should be identity."""
        cal = calibrate_simple(simple_axis_info, plot_area)
        test_pts = [(100.0, 100.0), (200.0, 200.0), (300.0, 150.0), (450.0, 330.0)]

        data_pts = pixel_to_data(test_pts, cal)
        back_pts = data_to_pixel(data_pts, cal)

        for (px, py), (px2, py2) in zip(test_pts, back_pts):
            assert abs(px2 - px) < 0.01, f"x mismatch: {px} → {px2}"
            assert abs(py2 - py) < 0.01, f"y mismatch: {py} → {py2}"


# ---------------------------------------------------------------------------
# calibrate_manual
# ---------------------------------------------------------------------------
class TestCalibrateManual:

    def test_two_ref_points(self, plot_area):
        x_refs = [
            {"pixel": 100.0, "value": 25.0},
            {"pixel": 300.0, "value": 75.0},
        ]
        y_refs = [
            {"pixel": 100.0, "value": 40.0},
            {"pixel": 200.0, "value": 20.0},
        ]
        cal = calibrate_manual(x_refs, y_refs, plot_area)
        assert cal.method == "manual"

        # Check that the x reference points map correctly
        pts = pixel_to_data([(100.0, 100.0)], cal)
        dx, dy = pts[0]
        assert abs(dx - 25.0) < 0.5
        assert abs(dy - 40.0) < 0.5


# ---------------------------------------------------------------------------
# calibrate_from_axis_info
# ---------------------------------------------------------------------------
class TestCalibrateFromAxisInfo:

    def test_simple_method(self, simple_axis_info, plot_area):
        cal = calibrate_from_axis_info(simple_axis_info, plot_area, method="simple")
        assert cal.method == "simple"

    def test_builds_mapping(self, simple_axis_info, plot_area):
        cal = calibrate_from_axis_info(simple_axis_info, plot_area)
        mapping = build_mapping_from_calibration(cal)
        assert mapping is not None
        assert mapping.pixel_to_data_matrix is not None


# ---------------------------------------------------------------------------
# validate_calibration
# ---------------------------------------------------------------------------
class TestValidateCalibration:

    def test_good_calibration(self, simple_axis_info, plot_area):
        cal = calibrate_simple(simple_axis_info, plot_area)
        test_pts = [(100.0, 100.0), (200.0, 200.0), (300.0, 150.0)]
        mean_err, p95_err = validate_calibration(test_pts, cal)
        assert mean_err < 1.0
        assert p95_err < 1.0

    def test_y_inversion(self, simple_axis_info, plot_area):
        """y increases downward in pixels, upward in data → check sign."""
        cal = calibrate_simple(simple_axis_info, plot_area)
        left, top, right, bottom = plot_area

        # Moving DOWN in pixel space should DECREASE data y
        pts_top = pixel_to_data([(float(left), float(top))], cal)
        pts_bot = pixel_to_data([(float(left), float(bottom))], cal)
        _, dy_top = pts_top[0]
        _, dy_bot = pts_bot[0]
        assert dy_top > dy_bot, "y should decrease as pixel-y increases"
