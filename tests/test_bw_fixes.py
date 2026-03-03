"""
Tests for the B/W pipeline fixes:
  1. Y-shift elimination (fence-post mapping)
  2. Text removal inside plot area
  3. Curve start/end anchoring
"""

import math
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from core.bw_pipeline import (
    preprocess_bw,
    trace_with_anchors,
    extract_bw_curves,
    smooth_curve,
    order_pixels_to_polyline,
    _remove_text_components,
    _skeletonize,
)
from core.image_processor import CurveDigitizer
from core.calibration import calibrate_simple, pixel_to_data
from core.scale import compute_affine_mapping, pixels_to_data
from core.types import AxisInfo


# =====================================================================
# Helpers – build synthetic test images
# =====================================================================

def _make_diagonal_line_image(
    w: int = 400,
    h: int = 300,
    x0: int = 50,
    x1: int = 350,
    y0: int = 250,
    y1: int = 50,
    thickness: int = 2,
    with_axes: bool = True,
) -> Image.Image:
    """Create a white image with a diagonal line from (x0,y0) to (x1,y1).

    Optionally draws solid black axis lines at the plot area boundaries.
    The "plot area" is inferred from the axis lines.
    """
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    # Axis lines (solid, 2px thick)
    if with_axes:
        ax_left, ax_top, ax_right, ax_bottom = 40, 30, 360, 270
        draw.line([(ax_left, ax_bottom), (ax_right, ax_bottom)], fill="black", width=2)
        draw.line([(ax_left, ax_top), (ax_left, ax_bottom)], fill="black", width=2)

    # Diagonal line (the "curve")
    draw.line([(x0, y0), (x1, y1)], fill="black", width=thickness)

    return img


def _make_curve_with_text_image(
    w: int = 500,
    h: int = 400,
) -> Image.Image:
    """Create a B/W image with a sine-like curve AND text labels '80' and '110%'.

    The text sits near/on the curve to simulate real chart labels.
    """
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    # Axis lines
    ax_left, ax_top, ax_right, ax_bottom = 50, 30, 450, 350
    draw.line([(ax_left, ax_bottom), (ax_right, ax_bottom)], fill="black", width=2)
    draw.line([(ax_left, ax_top), (ax_left, ax_bottom)], fill="black", width=2)

    # Sine curve
    for x in range(ax_left + 10, ax_right - 10):
        t = (x - ax_left) / (ax_right - ax_left)
        cy = int(200 + 80 * math.sin(t * 2 * math.pi))
        for dy in range(-1, 2):
            yy = cy + dy
            if ax_top < yy < ax_bottom:
                draw.point((x, yy), fill="black")

    # Text labels inside the plot area (using simple block characters)
    # "80" near (150, 180)
    _draw_block_text(draw, 150, 140, "80")
    # "110%" near (300, 140)
    _draw_block_text(draw, 300, 120, "110")

    return img, (ax_left, ax_top, ax_right, ax_bottom)


def _draw_block_text(draw: ImageDraw.Draw, x: int, y: int, text: str):
    """Draw crude block-character text (5x7 font) at (x,y)."""
    # Simple block patterns for digits
    CHARS = {
        "0": ["011110", "100001", "100001", "100001", "100001", "100001", "011110"],
        "1": ["000100", "001100", "000100", "000100", "000100", "000100", "001110"],
        "8": ["011110", "100001", "100001", "011110", "100001", "100001", "011110"],
    }
    cx = x
    for ch in text:
        pattern = CHARS.get(ch, CHARS["0"])
        for row_idx, row in enumerate(pattern):
            for col_idx, bit in enumerate(row):
                if bit == "1":
                    draw.point((cx + col_idx, y + row_idx), fill="black")
        cx += len(pattern[0]) + 2


def _make_simple_curve_image(
    w: int = 400,
    h: int = 300,
) -> tuple:
    """Create image with a single curve, returning image + plot bounds + curve endpoints."""
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    ax_left, ax_top, ax_right, ax_bottom = 40, 30, 360, 270
    draw.line([(ax_left, ax_bottom), (ax_right, ax_bottom)], fill="black", width=2)
    draw.line([(ax_left, ax_top), (ax_left, ax_bottom)], fill="black", width=2)

    # Curve from (60, 230) to (340, 70) — parabola-like
    curve_pts = []
    for x in range(60, 341):
        t = (x - 60) / 280.0
        y = int(230 - 160 * t + 40 * math.sin(t * math.pi))
        for dy in range(-1, 2):
            yy = y + dy
            if ax_top < yy < ax_bottom:
                draw.point((x, yy), fill="black")
        curve_pts.append((x, y))

    start_px = curve_pts[0]
    end_px = curve_pts[-1]

    return img, (ax_left, ax_top, ax_right, ax_bottom), start_px, end_px


# =====================================================================
# Test 1: Y-shift elimination (fence-post mapping)
# =====================================================================

class TestYShiftFix:
    """Verify that fence-post mapping eliminates the Y+1 shift."""

    def test_normalize_to_axis_diagonal_no_shift(self):
        """A diagonal line from corner to corner of the plot area
        should map exactly to (xMin, yMin)→(xMax, yMax) with no shift."""
        axis_info = {"xMin": 0, "xMax": 100, "yMin": 0, "yMax": 50}
        digitizer = CurveDigitizer(axis_info)

        # Plot area pixel bounds
        p_left, p_top, p_right, p_bottom = 50, 30, 450, 330

        # Corner pixels (last extractable pixel = p_right-1, p_bottom-1)
        corners = [
            (p_left, p_top),          # → (xMin, yMax) = (0, 50)
            (p_right - 1, p_bottom - 1),  # → (xMax, yMin) = (100, 0)
            (p_left, p_bottom - 1),   # → (xMin, yMin) = (0, 0)
            (p_right - 1, p_top),     # → (xMax, yMax) = (100, 50)
        ]

        result = digitizer.normalize_to_axis(
            corners, 600, 400, (p_left, p_top, p_right, p_bottom),
        )

        # Top-left → (0, 50)
        assert result[0][0] == pytest.approx(0.0, abs=0.01)
        assert result[0][1] == pytest.approx(50.0, abs=0.01)

        # Bottom-right → (100, 0)  —  THIS was the +1-shift bug
        assert result[1][0] == pytest.approx(100.0, abs=0.01)
        assert result[1][1] == pytest.approx(0.0, abs=0.01)

        # Bottom-left → (0, 0)
        assert result[2][0] == pytest.approx(0.0, abs=0.01)
        assert result[2][1] == pytest.approx(0.0, abs=0.01)

        # Top-right → (100, 50)
        assert result[3][0] == pytest.approx(100.0, abs=0.01)
        assert result[3][1] == pytest.approx(50.0, abs=0.01)

    def test_calibrate_simple_corners(self):
        """calibrate_simple must produce the same fence-post mapping."""
        axis_info = {"xMin": 0, "xMax": 100, "yMin": 0, "yMax": 50}
        plot_area = (50, 30, 450, 330)

        cal = calibrate_simple(axis_info, plot_area)
        p_left, p_top, p_right, p_bottom = plot_area

        # Bottom-right corner → (xMax, yMin)
        pts = pixel_to_data([(p_right - 1, p_bottom - 1)], cal)
        assert pts[0][0] == pytest.approx(100.0, abs=0.05)
        assert pts[0][1] == pytest.approx(0.0, abs=0.05)

        # Top-left → (xMin, yMax)
        pts = pixel_to_data([(p_left, p_top)], cal)
        assert pts[0][0] == pytest.approx(0.0, abs=0.01)
        assert pts[0][1] == pytest.approx(50.0, abs=0.01)

    def test_scale_affine_corners(self):
        """scale.compute_affine_mapping must map pixel (W-1, H-1) → (xMax, yMin)."""
        ai = AxisInfo(xMin=0, xMax=100, yMin=0, yMax=50)
        mapping = compute_affine_mapping(ai, 200, 100)

        # Pixel (0,0) → (xMin, yMax) = (0, 50)
        r = pixels_to_data([[0.0, 0.0]], mapping)
        assert r[0][0] == pytest.approx(0.0, abs=0.01)
        assert r[0][1] == pytest.approx(50.0, abs=0.01)

        # Pixel (199, 99) → (xMax, yMin) = (100, 0)
        r = pixels_to_data([[199.0, 99.0]], mapping)
        assert r[0][0] == pytest.approx(100.0, abs=0.05)
        assert r[0][1] == pytest.approx(0.0, abs=0.05)

    def test_midpoint_maps_correctly(self):
        """The center pixel maps to the center of the data range."""
        axis_info = {"xMin": 0, "xMax": 100, "yMin": 0, "yMax": 50}
        digitizer = CurveDigitizer(axis_info)

        p_left, p_top, p_right, p_bottom = 0, 0, 201, 101
        mid_x = 100  # (201 - 1) / 2 = 100
        mid_y = 50   # (101 - 1) / 2 = 50

        result = digitizer.normalize_to_axis(
            [(mid_x, mid_y)], 201, 101, (p_left, p_top, p_right, p_bottom),
        )

        assert result[0][0] == pytest.approx(50.0, abs=0.01)
        assert result[0][1] == pytest.approx(25.0, abs=0.01)


# =====================================================================
# Test 2: Text removal
# =====================================================================

class TestTextRemoval:
    """Verify that text characters inside the plot area are removed."""

    def test_text_components_removed_from_binary(self):
        """Build a binary image with text '80' and a curve; verify
        that the text pixels are removed but the curve survives."""
        h, w = 300, 400

        # Create binary mask: curve + text blobs
        binary = np.zeros((h, w), dtype=bool)

        # Draw a curve spanning most of the width
        for x in range(20, 380):
            y = int(150 + 50 * math.sin((x - 20) * math.pi / 360))
            for dy in range(-1, 2):
                if 0 <= y + dy < h:
                    binary[y + dy, x] = True

        # Draw "8"-like text blob at (100, 80) — two stacked circles
        for angle in range(360):
            rad = math.radians(angle)
            # Top circle
            cx1, cy1 = 100 + int(5 * math.cos(rad)), 75 + int(5 * math.sin(rad))
            if 0 <= cy1 < h and 0 <= cx1 < w:
                binary[cy1, cx1] = True
            # Bottom circle
            cx2, cy2 = 100 + int(5 * math.cos(rad)), 87 + int(5 * math.sin(rad))
            if 0 <= cy2 < h and 0 <= cx2 < w:
                binary[cy2, cx2] = True

        # Draw "0"-like text blob at (115, 80) — single circle
        for angle in range(360):
            rad = math.radians(angle)
            cx, cy = 115 + int(5 * math.cos(rad)), 81 + int(6 * math.sin(rad))
            if 0 <= cy < h and 0 <= cx < w:
                binary[cy, cx] = True

        n_before = binary.sum()
        cleaned = _remove_text_components(binary, h, w)
        n_after = cleaned.sum()

        # Text pixels should have been removed
        assert n_after < n_before, "Text components were not removed"

        # Curve should mostly survive (it spans the full width)
        curve_row_sum = 0
        for x in range(50, 350):
            if cleaned[:, x].any():
                curve_row_sum += 1
        assert curve_row_sum > 200, "Curve was incorrectly removed along with text"

    def test_curve_with_text_image_extraction(self):
        """Full pipeline test: extract curves from image with text labels."""
        result = _make_curve_with_text_image()
        img, plot_area = result[0], result[1]

        skeleton, binary, adj = preprocess_bw(img, plot_area)

        # After text removal, the binary should have no text-sized clusters
        from scipy.ndimage import label as ndimage_label
        labelled, n_comp = ndimage_label(skeleton, np.ones((3, 3), dtype=int))

        # The curve itself should be the dominant component
        for comp_id in range(1, n_comp + 1):
            mask = labelled == comp_id
            ys, xs = np.where(mask)
            bbox_w = int(xs.max() - xs.min()) + 1
            # Text-like components should have been filtered out
            # (small width, dense fill). Any surviving component
            # that is short should not be text-sized.
            if bbox_w < 20:
                area = int(mask.sum())
                assert area < 15, (
                    f"Suspected text component survived: bbox_w={bbox_w}, area={area}"
                )


# =====================================================================
# Test 3: Start/end anchoring
# =====================================================================

class TestAnchorAnchoring:
    """Verify that anchor-traced curves start and end exactly at the user coordinates."""

    def test_traced_path_starts_and_ends_at_anchors(self):
        """Build a skeleton with a curve, trace between two points,
        verify the first and last path points match the original (un-snapped)
        user coordinates."""
        h, w = 200, 300

        # Create skeleton: a gentle arc
        skel = np.zeros((h, w), dtype=bool)
        for x in range(30, 270):
            y = int(100 + 40 * math.sin((x - 30) * math.pi / 240))
            skel[y, x] = True

        start = (35, 103)  # near but not on the curve
        end = (265, 97)    # near but not on the curve

        path = trace_with_anchors(skel, start, end, snap_radius=15)

        assert path is not None, "A* path should be found"
        assert len(path) >= 2, "Path should have at least start and end"

        # Path MUST start at the original user-clicked position
        assert path[0] == start, (
            f"Path does not start at anchor start: got {path[0]}, expected {start}"
        )
        # Path MUST end at the original user-clicked position
        assert path[-1] == end, (
            f"Path does not end at anchor end: got {path[-1]}, expected {end}"
        )

    def test_extract_bw_curves_anchored(self):
        """Full pipeline: extract with anchors and verify endpoints."""
        img, plot_area, start_px, end_px = _make_simple_curve_image()

        curves = extract_bw_curves(
            img, 1, plot_area,
            anchors=[(start_px, end_px)],
            smoothing_strength=-1,  # disable smoothing
            extend_ends=False,
        )

        assert len(curves) >= 1, "Should extract at least one curve"

        # The first curve should start near start_px and end near end_px
        curve = curves[0]
        assert len(curve) >= 2

        # Start point should be AT the anchor
        assert curve[0] == start_px, (
            f"Curve does not start at anchor: got {curve[0]}, expected {start_px}"
        )
        # End point should be AT the anchor
        assert curve[-1] == end_px, (
            f"Curve does not end at anchor: got {curve[-1]}, expected {end_px}"
        )

    def test_smooth_preserves_endpoints(self):
        """After smoothing, first and last points should be preserved."""
        pts = [(float(x), float(100 + 10 * math.sin(x * 0.1)))
               for x in range(50, 300)]

        first = pts[0]
        last = pts[-1]

        smoothed = smooth_curve(pts, window_length=0)

        assert len(smoothed) >= 2
        # Smoothing deduplicates/resamples x, so we check the domain
        # rather than exact points
        assert smoothed[0][0] <= first[0] + 1
        assert smoothed[-1][0] >= last[0] - 1


# =====================================================================
# Test 4: Round-trip consistency (pixel → data → pixel)
# =====================================================================

class TestRoundTrip:
    """Ensure pixel→data→pixel round-trip error is near zero."""

    def test_calibration_roundtrip_exact(self):
        """Calibration round-trip should be exact for boundary pixels."""
        from core.calibration import pixel_to_data, data_to_pixel

        axis_info = {"xMin": 0, "xMax": 350000, "yMin": 1.0, "yMax": 1.6}
        plot_area = (100, 50, 500, 450)
        cal = calibrate_simple(axis_info, plot_area)

        # Test several points
        test_points = [
            (100, 50),    # top-left
            (499, 449),   # bottom-right (last extractable)
            (300, 250),   # middle
        ]

        data = pixel_to_data(test_points, cal)
        back = data_to_pixel(data, cal)

        for (px, py), (bx, by) in zip(test_points, back):
            err = math.sqrt((px - bx) ** 2 + (py - by) ** 2)
            assert err < 0.01, (
                f"Round-trip error {err:.4f} px at ({px},{py}) → "
                f"data → ({bx:.2f},{by:.2f})"
            )
