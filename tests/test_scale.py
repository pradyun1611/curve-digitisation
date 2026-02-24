"""
Unit tests for core.scale module (affine mapping + round-trip error).
"""

import numpy as np
import pytest

from core.scale import compute_affine_mapping, pixels_to_data, data_to_pixels, roundtrip_error
from core.types import AxisInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
AXIS_DICT = {
    "xMin": 0, "xMax": 100,
    "yMin": 0, "yMax": 50,
    "xUnit": "%", "yUnit": "units",
}


def _make_axis_info() -> AxisInfo:
    return AxisInfo.from_dict(AXIS_DICT)


class TestComputeAffineMapping:
    """Tests for compute_affine_mapping."""

    def test_returns_mapping_result(self):
        mapping = compute_affine_mapping(_make_axis_info(), 200, 100)
        assert mapping is not None
        assert mapping.plot_area_width == 200
        assert mapping.plot_area_height == 100
        assert mapping.pixel_to_data_matrix is not None
        assert mapping.data_to_pixel_matrix is not None

    def test_origin_maps_to_xmin_ymax(self):
        """Pixel (0,0) = top-left should map to (xMin, yMax)."""
        mapping = compute_affine_mapping(_make_axis_info(), 200, 100)
        result = pixels_to_data([[0.0, 0.0]], mapping)
        assert result[0][0] == pytest.approx(0.0, abs=1e-3)   # xMin
        assert result[0][1] == pytest.approx(50.0, abs=1e-3)  # yMax

    def test_bottom_right_maps_to_xmax_ymin(self):
        """Pixel (W-1, H-1) ~ (xMax, yMin)."""
        mapping = compute_affine_mapping(_make_axis_info(), 200, 100)
        result = pixels_to_data([[199.0, 99.0]], mapping)
        # Should be close to (100, 0)
        assert result[0][0] == pytest.approx(100.0, abs=1.0)
        assert result[0][1] == pytest.approx(0.0, abs=1.0)

    def test_center_maps_to_midpoints(self):
        """Pixel (100, 50) should map to (50, 25) with 200x100 image."""
        mapping = compute_affine_mapping(_make_axis_info(), 200, 100)
        result = pixels_to_data([[100.0, 50.0]], mapping)
        assert result[0][0] == pytest.approx(50.0, abs=1.0)
        assert result[0][1] == pytest.approx(25.0, abs=1.0)


class TestRoundTrip:
    """Test pixel -> data -> pixel round-trip."""

    def test_roundtrip_zero_error(self):
        """Round-trip should produce near-zero error for exact affine mapping."""
        mapping = compute_affine_mapping(_make_axis_info(), 200, 100)
        pts = [[50.0, 25.0], [100.0, 50.0], [150.0, 75.0]]
        mean_err, p95_err = roundtrip_error(pts, mapping)
        assert mean_err < 0.01
        assert p95_err < 0.01

    def test_roundtrip_returns_two_floats(self):
        mapping = compute_affine_mapping(_make_axis_info(), 200, 100)
        mean_err, p95_err = roundtrip_error([[10.0, 10.0]], mapping)
        assert isinstance(mean_err, float)
        assert isinstance(p95_err, float)


class TestDataToPixels:
    """Test data_to_pixels inverse mapping."""

    def test_inverse_of_forward(self):
        """data_to_pixels should be the inverse of pixels_to_data."""
        mapping = compute_affine_mapping(_make_axis_info(), 200, 100)
        px_in = [[30.0, 60.0], [150.0, 20.0]]
        data = pixels_to_data(px_in, mapping)
        px_out = data_to_pixels(data, mapping)
        for i in range(len(px_in)):
            assert px_out[i][0] == pytest.approx(px_in[i][0], abs=0.01)
            assert px_out[i][1] == pytest.approx(px_in[i][1], abs=0.01)
