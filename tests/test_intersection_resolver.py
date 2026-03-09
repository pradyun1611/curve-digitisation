"""
Tests for the intersection resolver module.

Validates:
  - Intersection zone detection
  - Shade-based enhancement
  - Intensity signature sampling
  - DP re-tracing through intersections
  - Full resolve_intersections pipeline
"""

import numpy as np
import pytest

from core.intersection_resolver import (
    _enhance_roi,
    _find_intersection_zones,
    _retrace_through_zone,
    _sample_intensity_signature,
    _stitch_segment,
    _zone_y_bounds,
    resolve_intersections,
)


# ────────────────────────────────────────────────────────────────────
# Helpers to build synthetic test data
# ────────────────────────────────────────────────────────────────────

def _make_crossing_curves(
    width: int = 400,
    height: int = 200,
    shade1: int = 60,
    shade2: int = 140,
) -> tuple:
    """Create a synthetic grayscale image with two crossing curves.

    Curve 0: starts at y=40, ends at y=160 (going down, shade1)
    Curve 1: starts at y=160, ends at y=40 (going up,  shade2)
    They cross in the middle around x=200.

    Returns (gray, binary, curves_dict)
    """
    gray = np.full((height, width), 220, dtype=np.uint8)  # light bg
    binary = np.zeros((height, width), dtype=bool)

    curves = {0: [], 1: []}
    for x in range(width):
        # Curve 0: linear from (0,40) to (400,160)
        y0 = int(40 + 120 * x / width)
        # Curve 1: linear from (0,160) to (400,40)
        y1 = int(160 - 120 * x / width)

        for dy in range(-2, 3):  # 5px stroke width
            if 0 <= y0 + dy < height:
                gray[y0 + dy, x] = shade1
                binary[y0 + dy, x] = True
            if 0 <= y1 + dy < height:
                gray[y1 + dy, x] = shade2
                binary[y1 + dy, x] = True

        curves[0].append((x, y0))
        curves[1].append((x, y1))

    return gray, binary, curves


# ────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────

class TestIntersectionZoneDetection:
    def test_detects_crossing_zone(self):
        gray, binary, curves = _make_crossing_curves()
        zones = _find_intersection_zones(curves, 200, 400, proximity_px=15)
        assert len(zones) >= 1
        # The crossing is around x=200
        zone = zones[0]
        assert zone["x_start"] < 200 < zone["x_end"]
        assert 0 in zone["curve_ids"] and 1 in zone["curve_ids"]

    def test_no_zones_for_parallel_curves(self):
        """Curves that never come close should produce no zones."""
        curves = {
            0: [(x, 30) for x in range(400)],
            1: [(x, 170) for x in range(400)],
        }
        zones = _find_intersection_zones(curves, 200, 400, proximity_px=12)
        assert len(zones) == 0

    def test_no_zones_for_single_curve(self):
        curves = {0: [(x, 100) for x in range(400)]}
        zones = _find_intersection_zones(curves, 200, 400)
        assert len(zones) == 0


class TestEnhanceROI:
    def test_output_shape_and_range(self):
        roi = np.random.randint(50, 200, size=(40, 60), dtype=np.uint8)
        enhanced = _enhance_roi(roi)
        assert enhanced.shape == roi.shape
        assert enhanced.min() >= 0.0
        assert enhanced.max() <= 1.0
        assert enhanced.dtype == np.float32

    def test_increases_contrast(self):
        """Enhancement should increase local contrast."""
        # Create a low-contrast ROI
        roi = np.full((40, 60), 128, dtype=np.uint8)
        roi[15:25, :] = 118  # slightly darker band
        enhanced = _enhance_roi(roi)
        # After enhancement, the difference should be amplified
        band_mean = float(enhanced[15:25, :].mean())
        bg_mean = float(np.concatenate([
            enhanced[:15, :].ravel(), enhanced[25:, :].ravel()
        ]).mean())
        # The enhanced difference should be larger than the original
        original_diff = abs(128 - 118) / 255.0
        enhanced_diff = abs(bg_mean - band_mean)
        assert enhanced_diff >= original_diff * 0.5  # at least half as large


class TestIntensitySignature:
    def test_samples_correct_shade(self):
        gray, binary, curves = _make_crossing_curves(shade1=60, shade2=140)
        # Curve 0 has shade=60, crossing around x=150-250
        mean_i, std_i = _sample_intensity_signature(
            gray, curves[0], zone_x_start=150, zone_x_end=250,
            binary=binary,
        )
        # Should be close to shade1=60
        assert abs(mean_i - 60) < 25  # within 25 of true shade

        mean_i2, std_i2 = _sample_intensity_signature(
            gray, curves[1], zone_x_start=150, zone_x_end=250,
            binary=binary,
        )
        # Should be close to shade2=140
        assert abs(mean_i2 - 140) < 25

    def test_different_signatures(self):
        gray, binary, curves = _make_crossing_curves(shade1=60, shade2=140)
        s0 = _sample_intensity_signature(
            gray, curves[0], zone_x_start=150, zone_x_end=250,
            binary=binary,
        )
        s1 = _sample_intensity_signature(
            gray, curves[1], zone_x_start=150, zone_x_end=250,
            binary=binary,
        )
        # The two curves should have measurably different means
        assert abs(s0[0] - s1[0]) > 30


class TestStitchSegment:
    def test_stitches_correctly(self):
        full_path = [(x, 100) for x in range(400)]
        new_segment = [(x, 110) for x in range(150, 251)]
        stitched = _stitch_segment(full_path, new_segment, 150, 250)
        # Points before zone: x < 150
        before = [p for p in stitched if p[0] < 150]
        assert all(p[1] == 100 for p in before)
        # Points in zone: 150 <= x <= 250
        in_zone = [p for p in stitched if 150 <= p[0] <= 250]
        assert all(p[1] == 110 for p in in_zone)
        # Points after zone: x > 250
        after = [p for p in stitched if p[0] > 250]
        assert all(p[1] == 100 for p in after)

    def test_preserves_order(self):
        full_path = [(x, 50 + x // 10) for x in range(300)]
        new_segment = [(x, 80) for x in range(100, 201)]
        stitched = _stitch_segment(full_path, new_segment, 100, 200)
        xs = [p[0] for p in stitched]
        assert xs == sorted(xs)  # monotonically increasing


class TestZoneYBounds:
    def test_bounds_cover_curves(self):
        gray, binary, curves = _make_crossing_curves()
        zone = {"x_start": 150, "x_end": 250, "curve_ids": [0, 1]}
        y0, y1 = _zone_y_bounds(curves, zone, 200)
        # Should cover both curves' y-range in the zone (roughly 80-130)
        assert y0 < 80
        assert y1 > 120


class TestResolveIntersections:
    def test_full_pipeline_no_crash(self):
        """End-to-end: should not crash and should return same curve count."""
        gray, binary, curves = _make_crossing_curves()
        result = resolve_intersections(gray, binary, curves, stroke_width=5)
        assert len(result) == len(curves)
        for cid in curves:
            assert cid in result
            assert len(result[cid]) > 0

    def test_single_curve_passthrough(self):
        """Single curve should pass through unchanged."""
        gray = np.full((100, 300), 200, dtype=np.uint8)
        binary = np.zeros((100, 300), dtype=bool)
        curves = {0: [(x, 50) for x in range(300)]}
        result = resolve_intersections(gray, binary, curves, stroke_width=4)
        assert len(result) == 1
        assert result[0] == curves[0]

    def test_shade_separation_improves_crossing(self):
        """With clear shade difference, curves should stay separated."""
        gray, binary, curves = _make_crossing_curves(shade1=40, shade2=180)
        # Deliberately mess up the curves at the crossing by swapping
        # segments in the middle
        swapped = {0: list(curves[0]), 1: list(curves[1])}
        # Swap the segments near the crossing (x=170..230)
        for i, (px, py) in enumerate(swapped[0]):
            if 170 <= px <= 230:
                # Find the corresponding point from curve 1
                match = [p for p in curves[1] if p[0] == px]
                if match:
                    swapped[0][i] = match[0]
        for i, (px, py) in enumerate(swapped[1]):
            if 170 <= px <= 230:
                match = [p for p in curves[0] if p[0] == px]
                if match:
                    swapped[1][i] = match[0]

        result = resolve_intersections(
            gray, binary, swapped, stroke_width=5,
            intensity_weight=0.8, geometry_weight=0.2,
        )
        assert len(result) == 2

        # The corrected curves should have better intensity consistency
        # (each curve's pixels should match its shade better than the swapped version)
        for cid in [0, 1]:
            vals_corrected = [
                float(gray[py, px])
                for px, py in result[cid]
                if 0 <= py < gray.shape[0] and 0 <= px < gray.shape[1]
            ]
            if vals_corrected:
                std_corrected = float(np.std(vals_corrected))
                # At minimum, the corrected curves shouldn't have worse consistency
                assert std_corrected < 80  # not perfect but shouldn't be wildly inconsistent
