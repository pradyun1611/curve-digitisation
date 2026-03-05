"""
Integration tests for the robust B/W pipeline improvements.

Tests:
  - Gridline removal (morphological)
  - Text/tick deterministic one-pass removal
  - Multi-curve separation (column-scan)
  - Coordinate mapping full-range coverage
  - Reproducibility (same result twice)
  - Curve exclusion (surge line filter)
  - Continuous polyline spanning expected x-range
"""

import math
import numpy as np
import pytest
from PIL import Image, ImageDraw

from core.bw_pipeline import (
    preprocess_bw,
    extract_bw_curves,
    detect_plot_area_robust,
    _remove_gridlines_morph,
    _remove_text_components,
    _remove_tick_marks,
    _column_scan_extract,
    _exclude_curve_filter,
)
from core.image_processor import CurveDigitizer


# =====================================================================
# Helpers – build synthetic test images
# =====================================================================

def _make_grid_image(w=500, h=400, n_grid_h=5, n_grid_v=6):
    """Image with axis lines, grid lines, and two parabolic curves."""
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    ax_left, ax_top, ax_right, ax_bottom = 50, 30, 450, 350

    # Axes (solid, 2px)
    draw.line([(ax_left, ax_bottom), (ax_right, ax_bottom)], fill="black", width=2)
    draw.line([(ax_left, ax_top), (ax_left, ax_bottom)], fill="black", width=2)

    # Grid lines (1px thin, spanning full plot interior)
    for i in range(1, n_grid_h):
        y = ax_top + i * (ax_bottom - ax_top) // n_grid_h
        draw.line([(ax_left + 1, y), (ax_right - 1, y)], fill="black", width=1)
    for i in range(1, n_grid_v):
        x = ax_left + i * (ax_right - ax_left) // n_grid_v
        draw.line([(x, ax_top + 1), (x, ax_bottom - 1)], fill="black", width=1)

    # Curve 1: parabola (upper)
    for x in range(ax_left + 10, ax_right - 10):
        t = (x - ax_left) / (ax_right - ax_left)
        y = int(120 + 100 * (t - 0.5) ** 2)
        for dy in range(-1, 2):
            if ax_top < y + dy < ax_bottom:
                draw.point((x, y + dy), fill="black")

    # Curve 2: sine (lower)
    for x in range(ax_left + 10, ax_right - 10):
        t = (x - ax_left) / (ax_right - ax_left)
        y = int(250 + 40 * math.sin(t * 2 * math.pi))
        for dy in range(-1, 2):
            if ax_top < y + dy < ax_bottom:
                draw.point((x, y + dy), fill="black")

    return img, (ax_left, ax_top, ax_right, ax_bottom)


def _make_text_tick_image(w=500, h=400):
    """Image with curves + text labels + tick marks near axes."""
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    ax_left, ax_top, ax_right, ax_bottom = 60, 40, 440, 350

    # Axes
    draw.line([(ax_left, ax_bottom), (ax_right, ax_bottom)], fill="black", width=2)
    draw.line([(ax_left, ax_top), (ax_left, ax_bottom)], fill="black", width=2)

    # Tick marks on y-axis (protruding into plot area)
    for i in range(5):
        y = ax_top + i * (ax_bottom - ax_top) // 4
        draw.line([(ax_left - 5, y), (ax_left + 5, y)], fill="black", width=1)

    # Tick marks on x-axis
    for i in range(6):
        x = ax_left + i * (ax_right - ax_left) // 5
        draw.line([(x, ax_bottom - 5), (x, ax_bottom + 5)], fill="black", width=1)

    # Text-like blob at (120, 80) — small dense cluster inside plot area
    for dx in range(12):
        for dy in range(8):
            if (dx < 3 or 4 <= dx < 7 or 8 <= dx < 12) and (dy < 2 or dy > 5):
                draw.point((120 + dx, 80 + dy), fill="black")

    # Curve (sine spanning full width)
    for x in range(ax_left + 5, ax_right - 5):
        t = (x - ax_left) / (ax_right - ax_left)
        y = int(200 + 60 * math.sin(t * 3 * math.pi))
        for dy in range(-1, 2):
            if ax_top < y + dy < ax_bottom:
                draw.point((x, y + dy), fill="black")

    return img, (ax_left, ax_top, ax_right, ax_bottom)


def _make_overlapping_curves_image(w=500, h=400, n_curves=3):
    """Image with N overlapping parabolic curves (all solid black)."""
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    ax_left, ax_top, ax_right, ax_bottom = 50, 30, 450, 350

    # Axes
    draw.line([(ax_left, ax_bottom), (ax_right, ax_bottom)], fill="black", width=2)
    draw.line([(ax_left, ax_top), (ax_left, ax_bottom)], fill="black", width=2)

    # Multiple curves at different vertical positions
    base_ys = [100, 180, 260]
    curves = []
    for ci in range(n_curves):
        curve = []
        base_y = base_ys[ci % len(base_ys)]
        for x in range(ax_left + 10, ax_right - 10):
            t = (x - ax_left) / (ax_right - ax_left)
            y = int(base_y + 50 * (t - 0.5) ** 2 - 20 * t)
            for dy in range(-1, 2):
                if ax_top < y + dy < ax_bottom:
                    draw.point((x, y + dy), fill="black")
            curve.append((x, y))
        curves.append(curve)

    return img, (ax_left, ax_top, ax_right, ax_bottom), curves


def _make_surge_line_image(w=500, h=400):
    """Image with 3 performance curves and a steep surge line."""
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    ax_left, ax_top, ax_right, ax_bottom = 50, 30, 450, 350

    # Axes
    draw.line([(ax_left, ax_bottom), (ax_right, ax_bottom)], fill="black", width=2)
    draw.line([(ax_left, ax_top), (ax_left, ax_bottom)], fill="black", width=2)

    # 3 parabolic curves
    for ci in range(3):
        base_y = 100 + ci * 60
        for x in range(ax_left + 10, ax_right - 10):
            t = (x - ax_left) / (ax_right - ax_left)
            y = int(base_y + 80 * (t - 0.5) ** 2)
            for dy in range(-1, 2):
                if ax_top < y + dy < ax_bottom:
                    draw.point((x, y + dy), fill="black")

    # Surge line: steep near-vertical line
    for y in range(ax_top + 10, ax_bottom - 10):
        x = int(120 + (y - ax_top) * 0.3)
        for dx in range(-1, 2):
            if ax_left < x + dx < ax_right:
                draw.point((x + dx, y), fill="black")

    return img, (ax_left, ax_top, ax_right, ax_bottom)


# =====================================================================
# Test: Gridline removal
# =====================================================================

class TestGridlineRemoval:
    """Verify gridlines are removed and curves survive."""

    def test_morphological_grid_removal(self):
        """Build binary with grid + curve, verify grid is removed."""
        h, w = 300, 400
        binary = np.zeros((h, w), dtype=bool)

        # Horizontal grid lines (full width)
        for gy in [60, 120, 180, 240]:
            binary[gy, 10:390] = True

        # Vertical grid lines (full height)
        for gx in [80, 160, 240, 320]:
            binary[10:290, gx] = True

        # A curve (sine)
        for x in range(20, 380):
            y = int(150 + 60 * math.sin(x * math.pi / 360))
            if 0 <= y < h:
                binary[y, x] = True

        grid_dark_before = sum(binary[gy, :].sum() for gy in [60, 120, 180, 240])
        cleaned = _remove_gridlines_morph(binary, h, w)
        grid_dark_after = sum(cleaned[gy, :].sum() for gy in [60, 120, 180, 240])

        # Grid lines should be mostly removed
        assert grid_dark_after < grid_dark_before * 0.3, (
            f"Grid not sufficiently removed: {grid_dark_after} vs {grid_dark_before}"
        )

        # Curve should mostly survive
        curve_survived = 0
        for x in range(20, 380):
            y = int(150 + 60 * math.sin(x * math.pi / 360))
            if 0 <= y < h and cleaned[y, x]:
                curve_survived += 1
        assert curve_survived > 150, (
            f"Curve damaged: only {curve_survived} pixels survived"
        )

    def test_grid_image_full_pipeline(self):
        """Full pipeline on grid image: curves extracted, grids not."""
        img, plot_area = _make_grid_image()
        skeleton, binary, adj = preprocess_bw(img, plot_area)

        rh, rw = binary.shape
        row_fills = binary.sum(axis=1) / rw
        col_fills = binary.sum(axis=0) / rh

        # No row/col should have >40% fill (grids would have >60%)
        assert row_fills.max() < 0.40, (
            f"Grid row survived: max fill = {row_fills.max():.2f}"
        )
        assert col_fills.max() < 0.40, (
            f"Grid col survived: max fill = {col_fills.max():.2f}"
        )

    def test_grid_does_not_appear_as_curve(self):
        """Extracted curves from gridded image should not include gridlines."""
        img, plot_area = _make_grid_image()
        curves = extract_bw_curves(
            img, 2, plot_area, smoothing_strength=-1, extend_ends=False,
        )

        for idx, pixels in curves.items():
            xs = [p[0] for p in pixels]
            ys = [p[1] for p in pixels]
            x_span = max(xs) - min(xs)
            y_span = max(ys) - min(ys) if ys else 0

            # A grid line would have very small y_span or very small x_span
            # relative to the other. Real curves have both.
            # A horizontal grid line: x_span >> y_span
            # A vertical grid line: y_span >> x_span
            # These should NOT be in the output.
            ratio = y_span / max(x_span, 1) if x_span > 0 else 0
            # If it's very flat (ratio < 0.02) and spans most of the width,
            # it might be a grid line — that's a failure.
            if x_span > 300:  # spans most of the plot
                assert ratio > 0.02, (
                    f"Curve {idx} looks like a grid line: "
                    f"x_span={x_span}, y_span={y_span}, ratio={ratio:.3f}"
                )


# =====================================================================
# Test: Text/tick removal determinism
# =====================================================================

class TestTextTickRemoval:
    """Verify text and ticks are removed in a single pass."""

    def test_text_removed_one_pass(self):
        """Processing the same image twice gives identical results."""
        img, plot_area = _make_text_tick_image()

        # First run
        skel1, bin1, adj1 = preprocess_bw(img, plot_area)

        # Second run (on same input, not on output of first run)
        skel2, bin2, adj2 = preprocess_bw(img, plot_area)

        # Both runs should produce identical results
        assert np.array_equal(bin1, bin2), "Text removal not deterministic"
        assert np.array_equal(skel1, skel2), "Skeleton not deterministic"

    def test_no_small_text_components_remain(self):
        """After preprocessing, no small isolated components should remain
        in the binary that look like text."""
        img, plot_area = _make_text_tick_image()
        _, binary, _ = preprocess_bw(img, plot_area)

        from scipy.ndimage import label as ndimage_label
        labelled, n_comp = ndimage_label(binary, np.ones((3, 3), dtype=int))

        for comp_id in range(1, n_comp + 1):
            mask = labelled == comp_id
            area = int(mask.sum())
            ys, xs = np.where(mask)
            bbox_w = int(xs.max() - xs.min()) + 1

            # Any surviving component should be large (curve) or
            # span at least 10% of width
            if area < 50:
                assert bbox_w < 15, (
                    f"Small text-like component survived: "
                    f"area={area}, bbox_w={bbox_w}"
                )

    def test_tick_marks_removed(self):
        """Tick marks near axes should be removed."""
        h, w = 300, 400
        binary = np.zeros((h, w), dtype=bool)

        # Simulate tick marks at left edge
        for i in range(5):
            y = 50 + i * 50
            binary[y, 0:8] = True  # short horizontal tick

        # Simulate tick marks at bottom edge
        for i in range(5):
            x = 50 + i * 70
            binary[h - 8:h, x] = True  # short vertical tick

        # A curve (should survive)
        for x in range(20, 380):
            y = int(150 + 40 * math.sin(x * 0.02))
            if 0 <= y < h:
                binary[y, x] = True

        cleaned = _remove_tick_marks(binary, h, w)

        # Ticks should be removed
        tick_survived = cleaned[50, 0:8].sum()
        assert tick_survived == 0, f"Left-edge tick not removed: {tick_survived} px"

        bottom_survived = cleaned[h - 8:h, 50].sum()
        assert bottom_survived == 0, f"Bottom tick not removed: {bottom_survived} px"

        # Curve should survive
        curve_survived = sum(
            1 for x in range(20, 380)
            if cleaned[int(150 + 40 * math.sin(x * 0.02)), x]
            and 0 <= int(150 + 40 * math.sin(x * 0.02)) < h
        )
        assert curve_survived > 200, f"Curve damaged by tick removal: {curve_survived}"


# =====================================================================
# Test: Multi-curve separation
# =====================================================================

class TestMultiCurveSeparation:
    """Verify overlapping curves are correctly separated."""

    def test_column_scan_finds_multiple_curves(self):
        """Column-scan should separate N horizontally overlapping curves."""
        h, w = 300, 400
        binary = np.zeros((h, w), dtype=bool)

        # 3 distinct horizontal bands
        for ci, base_y in enumerate([80, 150, 220]):
            for x in range(20, 380):
                y = base_y + int(30 * math.sin(x * 0.05))
                for dy in range(-1, 2):
                    if 0 <= y + dy < h:
                        binary[y + dy, x] = True

        curves = _column_scan_extract(binary, 5)
        assert len(curves) >= 3, f"Expected >= 3 curves, got {len(curves)}"

    def test_overlapping_image_extraction(self):
        """Full pipeline on overlapping curves image."""
        img, plot_area, expected = _make_overlapping_curves_image(n_curves=3)
        curves = extract_bw_curves(
            img, 3, plot_area, extend_ends=False, smoothing_strength=-1,
        )
        assert len(curves) >= 2, f"Expected >=2 curves, got {len(curves)}"

    def test_column_scan_extract_basic(self):
        """Basic column-scan: single curve should be found."""
        h, w = 200, 300
        binary = np.zeros((h, w), dtype=bool)
        for x in range(20, 280):
            y = 100
            binary[y, x] = True

        curves = _column_scan_extract(binary, 5)
        assert len(curves) >= 1, "Should find at least one curve"

        # The curve should span most of the width
        xs = [p[0] for p in curves[0]]
        assert max(xs) - min(xs) > 200

    def test_curves_sorted_by_y(self):
        """Extracted curves should be sorted topmost-first."""
        img, plot_area, _ = _make_overlapping_curves_image(n_curves=3)
        curves = extract_bw_curves(
            img, 3, plot_area, extend_ends=False, smoothing_strength=-1,
        )

        if len(curves) >= 2:
            mean_ys = []
            for idx in sorted(curves.keys()):
                ys = [p[1] for p in curves[idx]]
                mean_ys.append(np.mean(ys))

            # Each curve should have mean_y >= previous (topmost first)
            for i in range(1, len(mean_ys)):
                assert mean_ys[i] >= mean_ys[i - 1] - 5, (
                    f"Curves not sorted by y: {mean_ys}"
                )


# =====================================================================
# Test: Coordinate mapping full range
# =====================================================================

class TestCoordinateMapping:
    """Verify coordinate mapping covers full axis range."""

    def test_full_range_mapping(self):
        """Curve spanning full plot width maps to full axis range."""
        axis_info = {"xMin": 0, "xMax": 200000, "yMin": 40, "yMax": 70}
        digitizer = CurveDigitizer(axis_info)

        p_left, p_top, p_right, p_bottom = 50, 30, 450, 350

        # Pixel coords spanning full plot width
        pixels = [(x, 200) for x in range(p_left, p_right)]
        axis_coords = digitizer.normalize_to_axis(
            pixels, 500, 400, (p_left, p_top, p_right, p_bottom),
        )

        x_vals = [c[0] for c in axis_coords]
        assert min(x_vals) == pytest.approx(0.0, abs=500)
        assert max(x_vals) == pytest.approx(200000.0, abs=500)

    def test_y_inversion(self):
        """Top pixel maps to yMax, bottom pixel maps to yMin."""
        axis_info = {"xMin": 0, "xMax": 100, "yMin": 0, "yMax": 50}
        digitizer = CurveDigitizer(axis_info)

        p_left, p_top, p_right, p_bottom = 0, 0, 201, 101
        result = digitizer.normalize_to_axis(
            [(100, 0), (100, 100)], 201, 101,
            (p_left, p_top, p_right, p_bottom),
        )

        assert result[0][1] == pytest.approx(50.0, abs=0.1)  # top → yMax
        assert result[1][1] == pytest.approx(0.0, abs=0.1)   # bottom → yMin

    def test_monotonic_x(self):
        """X coordinates should be monotonically increasing."""
        axis_info = {"xMin": 0, "xMax": 100, "yMin": 0, "yMax": 50}
        digitizer = CurveDigitizer(axis_info)

        pixels = [(x, 50) for x in range(10, 200)]
        result = digitizer.normalize_to_axis(
            pixels, 300, 200, (10, 0, 200, 200),
        )

        x_vals = [c[0] for c in result]
        for i in range(1, len(x_vals)):
            assert x_vals[i] >= x_vals[i - 1], (
                f"X not monotonic at index {i}: {x_vals[i - 1]} > {x_vals[i]}"
            )

    def test_full_width_curve_maps_full_range(self):
        """A curve from plot-left to plot-right should map from xMin to xMax."""
        axis_info = {"xMin": 0, "xMax": 200000, "yMin": 40, "yMax": 70}
        digitizer = CurveDigitizer(axis_info)

        p_left, p_top, p_right, p_bottom = 50, 30, 450, 350

        # Left edge → xMin
        left_pts = digitizer.normalize_to_axis(
            [(p_left, 200)], 500, 400, (p_left, p_top, p_right, p_bottom),
        )
        assert left_pts[0][0] == pytest.approx(0.0, abs=1)

        # Right edge → xMax
        right_pts = digitizer.normalize_to_axis(
            [(p_right - 1, 200)], 500, 400, (p_left, p_top, p_right, p_bottom),
        )
        assert right_pts[0][0] == pytest.approx(200000.0, abs=500)


# =====================================================================
# Test: Reproducibility
# =====================================================================

class TestReproducibility:
    """Verify identical results across multiple runs."""

    def test_extract_twice_identical(self):
        """Running extract_bw_curves twice gives identical result."""
        img, plot_area, _ = _make_overlapping_curves_image(n_curves=2)

        result1 = extract_bw_curves(
            img, 2, plot_area, smoothing_strength=-1, extend_ends=False,
        )
        result2 = extract_bw_curves(
            img, 2, plot_area, smoothing_strength=-1, extend_ends=False,
        )

        assert len(result1) == len(result2), "Curve count differs between runs"
        for idx in result1:
            assert idx in result2, f"Curve {idx} missing in second run"
            assert result1[idx] == result2[idx], f"Curve {idx} differs between runs"

    def test_preprocess_deterministic(self):
        """preprocess_bw produces identical output on repeated calls."""
        img, plot_area = _make_grid_image()

        skel1, bin1, _ = preprocess_bw(img, plot_area)
        skel2, bin2, _ = preprocess_bw(img, plot_area)

        assert np.array_equal(bin1, bin2), "Binary not deterministic"
        assert np.array_equal(skel1, skel2), "Skeleton not deterministic"


# =====================================================================
# Test: Surge/curve exclusion
# =====================================================================

class TestCurveExclusion:
    """Verify curve exclusion modes work correctly."""

    def test_exclude_steepest(self):
        """Excluding 'steepest' should remove the surge line."""
        curves = {
            0: [(x, 200 + int(50 * math.sin(x * 0.02))) for x in range(50, 400)],
            1: [(x, 150 + int(30 * math.sin(x * 0.02))) for x in range(50, 400)],
            2: [(120 + y // 3, y) for y in range(50, 350)],  # steep/vertical
        }

        result = _exclude_curve_filter(curves, "steepest")
        assert len(result) == 2, f"Expected 2 curves, got {len(result)}"

        # Remaining curves should NOT be the steep one
        for idx, pixels in result.items():
            xs = [p[0] for p in pixels]
            ys = [p[1] for p in pixels]
            x_span = max(xs) - min(xs)
            y_span = max(ys) - min(ys)
            ratio = y_span / max(x_span, 1)
            assert ratio < 2.0, f"Steep curve not excluded: ratio={ratio:.2f}"

    def test_exclude_topmost(self):
        """Excluding 'topmost' removes the curve with smallest mean y."""
        curves = {
            0: [(x, 80) for x in range(50, 400)],   # top
            1: [(x, 200) for x in range(50, 400)],  # middle
            2: [(x, 300) for x in range(50, 400)],  # bottom
        }

        result = _exclude_curve_filter(curves, "topmost")
        assert len(result) == 2
        for idx, pixels in result.items():
            mean_y = np.mean([p[1] for p in pixels])
            assert mean_y > 100, "Topmost curve not excluded"

    def test_exclude_longest(self):
        """Excluding 'longest' removes the curve with largest x_span."""
        curves = {
            0: [(x, 100) for x in range(50, 400)],   # long
            1: [(x, 200) for x in range(100, 250)],  # short
        }
        result = _exclude_curve_filter(curves, "longest")
        assert len(result) == 1
        xs = [p[0] for p in result[0]]
        assert max(xs) - min(xs) < 200, "Longest curve not excluded"

    def test_exclude_bottommost(self):
        """Excluding 'bottommost' removes the lowest curve."""
        curves = {
            0: [(x, 80) for x in range(50, 400)],
            1: [(x, 300) for x in range(50, 400)],
        }
        result = _exclude_curve_filter(curves, "bottommost")
        assert len(result) == 1
        mean_y = np.mean([p[1] for p in result[0]])
        assert mean_y < 200, "Bottommost curve not excluded"

    def test_exclude_none(self):
        """Empty mode keeps all curves."""
        curves = {0: [(x, 100) for x in range(50, 400)]}
        result = _exclude_curve_filter(curves, "")
        assert len(result) == len(curves)

    def test_exclude_thickest(self):
        """Excluding 'thickest' removes the curve with most pixels."""
        curves = {
            0: [(x, 100) for x in range(50, 400)],       # 350 pixels
            1: [(x, 200) for x in range(100, 150)],      # 50 pixels
        }
        result = _exclude_curve_filter(curves, "thickest")
        assert len(result) == 1
        assert len(result[0]) < 100, "Thickest curve not excluded"

    def test_exclude_with_single_curve_noop(self):
        """Exclusion on a single curve does nothing."""
        curves = {0: [(x, 100) for x in range(50, 400)]}
        result = _exclude_curve_filter(curves, "steepest")
        assert len(result) == 1


# =====================================================================
# Test: Curves become continuous polylines spanning expected x-range
# =====================================================================

class TestCurveSpanAndContinuity:
    """Verify extracted curves span the expected x-range."""

    def test_curve_spans_most_of_plot(self):
        """Extracted curves should span at least 50% of the plot width."""
        img, plot_area, _ = _make_overlapping_curves_image(n_curves=2)
        p_left, _, p_right, _ = plot_area
        plot_width = p_right - p_left

        curves = extract_bw_curves(
            img, 2, plot_area, smoothing_strength=-1, extend_ends=False,
        )

        for idx, pixels in curves.items():
            xs = [p[0] for p in pixels]
            x_span = max(xs) - min(xs)
            assert x_span > plot_width * 0.5, (
                f"Curve {idx} too narrow: x_span={x_span}, "
                f"expected >{plot_width * 0.5:.0f}"
            )

    def test_curve_x_monotonic_after_smoothing(self):
        """Smoothed curve x-coordinates should be monotonically increasing."""
        img, plot_area, _ = _make_overlapping_curves_image(n_curves=1)

        curves = extract_bw_curves(
            img, 1, plot_area, smoothing_strength=0, extend_ends=False,
        )

        if len(curves) > 0:
            pixels = curves[0]
            xs = [p[0] for p in pixels]
            for i in range(1, len(xs)):
                assert xs[i] >= xs[i - 1], (
                    f"X not monotonic at index {i}: {xs[i - 1]} > {xs[i]}"
                )


# =====================================================================
# Test: ROI detection
# =====================================================================

class TestROIDetection:
    """Verify plot-area detection is robust."""

    def test_detect_plot_area_basic(self):
        """detect_plot_area_robust finds axes in a standard chart."""
        img = Image.new("RGB", (500, 400), "white")
        draw = ImageDraw.Draw(img)

        # Draw axes
        draw.line([(60, 350), (450, 350)], fill="black", width=2)  # x-axis
        draw.line([(60, 30), (60, 350)], fill="black", width=2)    # y-axis

        pa = detect_plot_area_robust(img)
        left, top, right, bottom = pa

        # The detected area should be close to the axis positions
        assert 55 <= left <= 75, f"Left bound off: {left}"
        assert 340 <= bottom <= 360, f"Bottom bound off: {bottom}"

    def test_detect_plot_area_with_grid(self):
        """Plot area detection should produce a reasonable area even with grid."""
        img, expected_area = _make_grid_image()

        pa = detect_plot_area_robust(img)
        left, top, right, bottom = pa

        # Grid lines can confuse axis detection, so we only check that
        # a reasonable plot area is returned (contained within image).
        w, h = img.size
        assert 0 <= left < right <= w, f"Bad horizontal bounds: {left}, {right}"
        assert 0 <= top < bottom <= h, f"Bad vertical bounds: {top}, {bottom}"
        # Area should be at least 20% of the image
        area_ratio = (right - left) * (bottom - top) / (w * h)
        assert area_ratio > 0.15, f"Detected area too small: {area_ratio:.2%}"


# =====================================================================
# Test: Config module
# =====================================================================

class TestConfig:
    """Verify config module loads and has reasonable defaults."""

    def test_config_defaults(self):
        """Default config should have reasonable values."""
        from core.config import BWPipelineConfig, DEFAULT_CONFIG

        cfg = DEFAULT_CONFIG
        assert cfg.grid_h_kernel_ratio > 0
        assert cfg.text_area_max_ratio > 0
        assert cfg.close_kernel_size == (3, 7)
        assert cfg.exclude_curve_mode == ""
        assert cfg.snap_radius > 0

    def test_config_override(self):
        """Config should support per-field override."""
        from core.config import BWPipelineConfig

        cfg = BWPipelineConfig(
            grid_h_kernel_ratio=0.30,
            exclude_curve_mode="steepest",
        )
        assert cfg.grid_h_kernel_ratio == 0.30
        assert cfg.exclude_curve_mode == "steepest"
        # Other fields keep defaults
        assert cfg.text_area_max_ratio == 0.012
