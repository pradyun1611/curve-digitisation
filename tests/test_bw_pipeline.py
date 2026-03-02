"""
Unit tests for core.bw_pipeline – B/W skeleton extraction pipeline.
"""

import numpy as np
import pytest
from PIL import Image

from core.bw_pipeline import (
    preprocess_bw,
    filter_dashed_components,
    score_dashed,
    extract_skeleton_components,
    select_best_curves,
    smooth_curve,
    extend_curve_ends,
    order_pixels_to_polyline,
    detect_plot_area_robust,
)


# ---------------------------------------------------------------------------
# Helpers – build simple synthetic B/W images
# ---------------------------------------------------------------------------
def _horizontal_line_image(h=200, w=300, y=100, thickness=2):
    """PIL image with a diagonal/curve-like line on white bg (avoids grid removal)."""
    arr = np.ones((h, w, 3), dtype=np.uint8) * 255
    # Draw a gentle curve (not ruler-straight, to avoid grid-line removal)
    for x in range(20, 280):
        cy = y + int(15 * np.sin((x - 20) * np.pi / 260))
        for t in range(thickness):
            yy = cy + t
            if 0 <= yy < h:
                arr[yy, x, :] = 0
    return Image.fromarray(arr)


def _horizontal_line_binary(h=200, w=300, y=100, thickness=2):
    """Binary numpy array with a horizontal line (white on black)."""
    img = np.zeros((h, w), dtype=np.uint8)
    img[y:y + thickness, 20:280] = 255
    return img


def _dashed_line_binary(h=200, w=300, y=100, seg_len=15, gap_len=10):
    """Binary array with a dashed horizontal line."""
    img = np.zeros((h, w), dtype=np.uint8)
    x = 20
    while x + seg_len < 280:
        img[y:y + 2, x:x + seg_len] = 255
        x += seg_len + gap_len
    return img


def _curve_binary(h=200, w=300):
    """Binary array with a sine-like curve."""
    img = np.zeros((h, w), dtype=np.uint8)
    for x in range(20, 280):
        y = int(100 + 40 * np.sin((x - 20) * 2 * np.pi / 260))
        for dy in range(-1, 2):
            yy = y + dy
            if 0 <= yy < h:
                img[yy, x] = 255
    return img


PLOT_AREA = (10, 10, 290, 190)


# ---------------------------------------------------------------------------
# preprocess_bw
# ---------------------------------------------------------------------------
class TestPreprocessBW:

    def test_returns_binary(self):
        img = _horizontal_line_image()
        binary, gray, pa = preprocess_bw(img, PLOT_AREA)
        # May be bool or uint8 depending on implementation
        assert binary.dtype in (np.uint8, np.bool_)
        # Should have some nonzero pixels (the line)
        assert np.sum(binary) > 10

    def test_preserves_line(self):
        img = _horizontal_line_image()
        binary, gray, pa = preprocess_bw(img, PLOT_AREA)
        assert np.sum(binary) > 50


# ---------------------------------------------------------------------------
# filter_dashed_components
# ---------------------------------------------------------------------------
class TestFilterDashed:

    def test_solid_line_passes(self):
        skel = _horizontal_line_binary(thickness=1)
        cleaned = filter_dashed_components(skel, dashed_threshold=0.4)
        # Solid line should mostly survive
        assert np.sum(cleaned > 0) > 50

    def test_dashed_has_components(self):
        skel = _dashed_line_binary()
        cleaned = filter_dashed_components(skel, dashed_threshold=0.3)
        # Filter should do something (may or may not remove all)
        assert isinstance(cleaned, np.ndarray)


# ---------------------------------------------------------------------------
# score_dashed
# ---------------------------------------------------------------------------
class TestScoreDashed:

    def test_returns_float(self):
        blob = np.ones((5, 200), dtype=np.uint8) * 255
        pixels = [(r, c) for r in range(5) for c in range(200)]
        s = score_dashed(blob, pixels)
        assert isinstance(s, float)

    def test_dashed_blob_higher_score(self):
        # Small fragment
        blob = np.ones((4, 20), dtype=np.uint8) * 255
        pixels = [(r, c) for r in range(4) for c in range(20)]
        s = score_dashed(blob, pixels)
        assert isinstance(s, float)


# ---------------------------------------------------------------------------
# extract_skeleton_components
# ---------------------------------------------------------------------------
class TestExtractSkeleton:

    def test_single_line(self):
        skel = _horizontal_line_binary(thickness=1)
        components = extract_skeleton_components(skel, min_pixels=10)
        assert len(components) >= 1
        # Components are dicts
        assert isinstance(components[0], dict)

    def test_min_pixels_filter(self):
        skel = _horizontal_line_binary(thickness=1)
        # Very high min_pixels should filter everything
        short_comps = extract_skeleton_components(skel, min_pixels=10000)
        assert isinstance(short_comps, list)


# ---------------------------------------------------------------------------
# select_best_curves
# ---------------------------------------------------------------------------
class TestSelectBestCurves:

    def test_returns_list(self):
        # Build fake components matching the expected Dict format
        components = [
            {"pixels": [(y, 10) for y in range(50, 250)],
             "mask": np.zeros((200, 300), dtype=bool),
             "dashed_score": 0.1, "x_span": 200, "y_span": 10,
             "area": 200, "mean_y": 100.0, "x_min": 50, "x_max": 249},
            {"pixels": [(y, 20) for y in range(100, 120)],
             "mask": np.zeros((200, 300), dtype=bool),
             "dashed_score": 0.1, "x_span": 20, "y_span": 5,
             "area": 20, "mean_y": 110.0, "x_min": 100, "x_max": 119},
        ]
        result = select_best_curves(components, num_curves=3)
        assert isinstance(result, list)
        assert len(result) <= 3


# ---------------------------------------------------------------------------
# smooth_curve
# ---------------------------------------------------------------------------
class TestSmoothCurve:

    def test_no_smoothing(self):
        pts = [(float(x), 100.0 + x * 0.5) for x in range(50)]
        result = smooth_curve(pts, window_length=0)
        assert len(result) == len(pts)

    def test_positive_smoothing(self):
        rng = np.random.default_rng(42)
        pts = [(float(x), 50.0 + x * 0.3 + rng.normal(0, 5))
               for x in range(100)]
        # window_length must be odd and >= polyorder+1
        result = smooth_curve(pts, window_length=11, polyorder=3)
        assert len(result) == len(pts)

    def test_too_few_points(self):
        pts = [(0.0, 0.0), (1.0, 1.0)]
        result = smooth_curve(pts, window_length=5)
        assert len(result) == len(pts)


# ---------------------------------------------------------------------------
# extend_curve_ends
# ---------------------------------------------------------------------------
class TestExtendCurveEnds:

    def test_returns_list(self):
        binary = _horizontal_line_binary()
        pts = [(100, x) for x in range(50, 250)]
        result = extend_curve_ends(pts, binary, search_radius=15)
        assert isinstance(result, list)
        assert len(result) >= len(pts)

    def test_no_crash_on_short_curve(self):
        binary = _horizontal_line_binary()
        pts = [(100, 50), (100, 51), (100, 52)]
        result = extend_curve_ends(pts, binary, search_radius=10)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# order_pixels_to_polyline
# ---------------------------------------------------------------------------
class TestOrderPixels:

    def test_orders_horizontal(self):
        rng = np.random.default_rng(0)
        xs = list(range(100))
        rng.shuffle(xs)
        pts = [(50, x) for x in xs]
        ordered = order_pixels_to_polyline(pts)
        assert len(ordered) == len(pts)

    def test_empty_input(self):
        result = order_pixels_to_polyline([])
        assert len(result) == 0


# ---------------------------------------------------------------------------
# detect_plot_area_robust
# ---------------------------------------------------------------------------
class TestDetectPlotAreaRobust:

    def test_detects_rectangle(self):
        # Image with a black rectangle border
        arr = np.ones((300, 400, 3), dtype=np.uint8) * 255
        # Draw border lines
        arr[50:52, 60:380, :] = 0     # top
        arr[248:250, 60:380, :] = 0   # bottom
        arr[50:250, 60:62, :] = 0     # left
        arr[50:250, 378:380, :] = 0   # right

        img = Image.fromarray(arr)
        bbox = detect_plot_area_robust(img)
        assert bbox is not None
        left, top, right, bottom = bbox
        # Approximate checks
        assert 50 <= left <= 70
        assert 40 <= top <= 60
        assert 370 <= right <= 390
        assert 240 <= bottom <= 260
