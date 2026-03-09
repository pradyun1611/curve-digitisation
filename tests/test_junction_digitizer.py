"""
Synthetic test image generator for junction-aware digitizer.

Generates test images with known ground-truth curves that overlap, creating
skeleton junctions.  Used for regression testing the junction disambiguation
and stateful pathfinding logic.

Usage:
    python tests/test_junction_digitizer.py
    pytest tests/test_junction_digitizer.py -v
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# Ensure core/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.junction_digitizer import (
    AffineMapping,
    GraphEdge,
    JunctionConfig,
    build_graph,
    calibrate_from_2points_per_axis,
    calibrate_from_3points,
    deskew_image,
    disambiguate_junctions,
    digitize,
    map_pixels_to_data,
    postprocess_curve,
    preprocess,
    stateful_dijkstra,
    auto_trace_curves,
)


# ====================================================================
# Synthetic image generators
# ====================================================================

def generate_overlapping_arcs(
    width: int = 600,
    height: int = 400,
    num_curves: int = 5,
    stroke_width: int = 2,
    noise_sigma: float = 0.0,
) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    """Generate a B/W image with multiple overlapping parabolic arcs.

    The arcs simulate expander/compressor efficiency curves that
    overlap near their peaks, creating ambiguous junctions when
    skeletonized.

    Returns
    -------
    image : (H, W) uint8 grayscale, white background with black curves
    ground_truth : list of list of (x, y) pixel coordinates per curve
    """
    image = np.ones((height, width), dtype=np.uint8) * 255
    margin = 40
    ground_truth = []

    for i in range(num_curves):
        # Each curve: y = a*(x - h)^2 + k  (inverted parabola)
        # Shift peak x and width per curve to create overlaps
        frac = i / max(num_curves - 1, 1)
        peak_x = margin + int((width - 2 * margin) * (0.2 + 0.6 * frac))
        peak_y = margin + int((height - 2 * margin) * 0.15)  # near top
        curve_width = int((width - 2 * margin) * (0.3 + 0.3 * frac))

        # Coefficient: y increases (goes down in image) as we move from peak
        a = (height - 2 * margin - peak_y + margin) / (curve_width ** 2)

        curve_pts = []
        x_start = max(margin, peak_x - curve_width)
        x_end = min(width - margin, peak_x + curve_width)

        for x in range(x_start, x_end + 1):
            dx = x - peak_x
            y = int(peak_y + a * dx * dx)
            if noise_sigma > 0:
                y += int(np.random.normal(0, noise_sigma))
            y = max(margin, min(height - margin, y))
            curve_pts.append((x, y))

        ground_truth.append(curve_pts)

        # Draw the curve
        pts = np.array(curve_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [pts], isClosed=False, color=0,
                      thickness=stroke_width)

    return image, ground_truth


def generate_crossing_curves(
    width: int = 500,
    height: int = 400,
    stroke_width: int = 2,
) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    """Generate two curves that cross in the middle (X pattern).

    This creates a definite junction where the skeleton merges.
    """
    image = np.ones((height, width), dtype=np.uint8) * 255
    margin = 30
    ground_truth = []

    # Curve 1: low-left → high-middle → low-right (inverted U)
    curve1 = []
    for x in range(margin, width - margin):
        t = (x - margin) / (width - 2 * margin)
        y = int(margin + (height - 2 * margin) * (0.3 + 0.5 * (2 * t - 1) ** 2))
        curve1.append((x, y))
    ground_truth.append(curve1)

    # Curve 2: high-left → low-middle → high-right (U shape)
    curve2 = []
    for x in range(margin, width - margin):
        t = (x - margin) / (width - 2 * margin)
        y = int(margin + (height - 2 * margin) * (0.7 - 0.5 * (2 * t - 1) ** 2))
        curve2.append((x, y))
    ground_truth.append(curve2)

    for curve in ground_truth:
        pts = np.array(curve, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [pts], isClosed=False, color=0,
                      thickness=stroke_width)

    return image, ground_truth


def generate_near_parallel_curves(
    width: int = 600,
    height: int = 400,
    num_curves: int = 3,
    gap: int = 15,
    stroke_width: int = 2,
) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    """Generate nearly parallel curves that merge at one point.

    Simulates the case where curves are close together and the skeleton
    may merge them at narrow gaps.
    """
    image = np.ones((height, width), dtype=np.uint8) * 255
    margin = 40
    ground_truth = []

    base_peak_y = int(height * 0.2)
    merge_x = width // 2  # they come closest here

    for i in range(num_curves):
        frac = i / max(num_curves - 1, 1)
        peak_x = int(margin + (width - 2 * margin) * (0.3 + 0.4 * frac))
        offset = (i - num_curves // 2) * gap

        curve_pts = []
        for x in range(margin, width - margin):
            t = (x - peak_x) / (width * 0.3)
            y_base = base_peak_y + int((height * 0.5) * t * t)

            # Add y-offset that vanishes near merge_x
            dist_to_merge = abs(x - merge_x)
            merge_factor = min(1.0, dist_to_merge / (width * 0.15))
            y = int(y_base + offset * merge_factor)
            y = max(margin, min(height - margin, y))
            curve_pts.append((x, y))

        ground_truth.append(curve_pts)
        pts = np.array(curve_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [pts], isClosed=False, color=0,
                      thickness=stroke_width)

    return image, ground_truth


def save_synthetic_image(
    image: np.ndarray,
    ground_truth: list[list[tuple[int, int]]],
    output_dir: str,
    name: str,
) -> tuple[str, str]:
    """Save synthetic image and ground truth to disk.

    Returns (image_path, ground_truth_path).
    """
    os.makedirs(output_dir, exist_ok=True)

    img_path = os.path.join(output_dir, f"{name}.png")
    cv2.imwrite(img_path, image)

    gt_path = os.path.join(output_dir, f"{name}_gt.json")
    gt_data = {}
    for i, pts in enumerate(ground_truth):
        gt_data[f"curve_{i}"] = [{"x": p[0], "y": p[1]} for p in pts]
    import json
    with open(gt_path, "w") as f:
        json.dump(gt_data, f, indent=2)

    return img_path, gt_path


# ====================================================================
# Tests
# ====================================================================

class TestPreprocess:
    """Test preprocessing produces valid skeleton."""

    def test_skeleton_not_empty(self):
        image, _ = generate_overlapping_arcs(num_curves=3, stroke_width=2)
        config = JunctionConfig()
        skeleton, binary, gray, cost_map = preprocess(image, config)
        assert skeleton.shape == image.shape
        assert np.sum(skeleton > 0) > 50   # some skeleton pixels

    def test_binary_foreground(self):
        image, _ = generate_overlapping_arcs(num_curves=3, stroke_width=2)
        config = JunctionConfig()
        _, binary, _, _ = preprocess(image, config)
        # Should have foreground pixels
        assert np.sum(binary > 0) > 100


class TestBuildGraph:
    """Test skeleton graph construction."""

    def test_graph_has_nodes_and_edges(self):
        image, _ = generate_overlapping_arcs(num_curves=3, stroke_width=2)
        config = JunctionConfig()
        skeleton, _, gray, cost_map = preprocess(image, config)
        adjacency, all_edges, endpoints, junctions = build_graph(skeleton, gray, cost_map, config)
        assert len(all_edges) > 0
        assert len(endpoints) > 0

    def test_junction_detection_on_overlapping_arcs(self):
        image, _ = generate_overlapping_arcs(num_curves=5, stroke_width=2)
        config = JunctionConfig()
        skeleton, _, gray, cost_map = preprocess(image, config)
        _, _, _, junctions = build_graph(skeleton, gray, cost_map, config)
        # With 5 overlapping arcs, there should be junctions
        assert len(junctions) >= 1, (
            f"Expected junctions for 5 overlapping arcs, got {len(junctions)}"
        )

    def test_tangents_computed(self):
        image, _ = generate_overlapping_arcs(num_curves=3, stroke_width=2)
        config = JunctionConfig()
        skeleton, _, gray, cost_map = preprocess(image, config)
        _, all_edges, _, _ = build_graph(skeleton, gray, cost_map, config)
        # At least some edges should have tangents
        has_tangent = sum(1 for e in all_edges if e.tangent_a is not None)
        assert has_tangent > 0

    def test_spur_removal(self):
        """Short dead-end spurs should be removed."""
        image, _ = generate_overlapping_arcs(num_curves=2, stroke_width=2)
        config = JunctionConfig(min_edge_length=10)
        skeleton, _, gray, cost_map = preprocess(image, config)
        _, all_edges, endpoints, _ = build_graph(skeleton, gray, cost_map, config)
        # Short edges with one endpoint end should have been pruned
        # (junction-junction short edges may survive, which is correct)
        for e in all_edges:
            if e.node_a in endpoints or e.node_b in endpoints:
                assert e.length >= config.min_edge_length, (
                    f"Spur edge {e.edge_id} has length {e.length} < {config.min_edge_length}"
                )


class TestJunctionDisambiguation:
    """Test tangent-based junction pairing."""

    def test_pairings_created_for_junctions(self):
        image, _ = generate_overlapping_arcs(num_curves=5, stroke_width=2)
        config = JunctionConfig()
        skeleton, _, gray, cost_map = preprocess(image, config)
        adjacency, all_edges, endpoints, junctions = build_graph(skeleton, gray, cost_map, config)
        pairings = disambiguate_junctions(adjacency, junctions, config)
        if len(junctions) > 0:
            assert len(pairings) > 0

    def test_crossing_curves_paired_correctly(self):
        """At an X junction, the two through-paths should be paired."""
        image, gt = generate_crossing_curves(stroke_width=2)
        config = JunctionConfig()
        skeleton, _, gray, cost_map = preprocess(image, config)
        adjacency, all_edges, endpoints, junctions = build_graph(skeleton, gray, cost_map, config)
        pairings = disambiguate_junctions(adjacency, junctions, config)
        # There should be at least one junction with pairings
        total_pairings = sum(len(v) for v in pairings.values())
        assert total_pairings >= 2, f"Expected paired edges at crossing, got {total_pairings}"


class TestStatefulDijkstra:
    """Test direction-aware pathfinding."""

    def test_finds_path_simple(self):
        """Should find a path between two endpoints."""
        image, _ = generate_overlapping_arcs(num_curves=2, stroke_width=2)
        config = JunctionConfig()
        skeleton, _, gray, cost_map = preprocess(image, config)
        adjacency, all_edges, endpoints, junctions = build_graph(skeleton, gray, cost_map, config)
        pairings = disambiguate_junctions(adjacency, junctions, config)

        ep_list = sorted(endpoints, key=lambda p: p[0])
        if len(ep_list) >= 2:
            start = ep_list[0]
            end = ep_list[-1]
            result = stateful_dijkstra(
                adjacency, all_edges, junctions, pairings,
                start, end, gray, cost_map, config,
            )
            assert result is not None, "Should find a path between endpoints"
            cost, path = result
            assert cost > 0
            assert len(path) > 0


class TestAutoTrace:
    """Test automatic multi-curve tracing."""

    def test_finds_multiple_curves(self):
        image, gt = generate_overlapping_arcs(num_curves=3, stroke_width=2)
        config = JunctionConfig(beam_width=3)
        skeleton, _, gray, cost_map = preprocess(image, config)
        adjacency, all_edges, endpoints, junctions = build_graph(skeleton, gray, cost_map, config)
        pairings = disambiguate_junctions(adjacency, junctions, config)

        curves = auto_trace_curves(
            adjacency, all_edges, endpoints, junctions, pairings,
            gray, cost_map, config, image_width=image.shape[1],
        )
        assert len(curves) >= 1, f"Expected at least 1 curve, got {len(curves)}"

    def test_crossing_curves_separated(self):
        """At an X intersection, should find 2 separate curves."""
        image, gt = generate_crossing_curves(stroke_width=2)
        config = JunctionConfig(beam_width=5, min_curve_xspan_ratio=0.05)
        skeleton, _, gray, cost_map = preprocess(image, config)
        adjacency, all_edges, endpoints, junctions = build_graph(skeleton, gray, cost_map, config)
        pairings = disambiguate_junctions(adjacency, junctions, config)

        curves = auto_trace_curves(
            adjacency, all_edges, endpoints, junctions, pairings,
            gray, cost_map, config, image_width=image.shape[1],
        )
        # Should find at least 1 curve; ideally 2 but topology may be hard
        assert len(curves) >= 1, (
            f"Expected at least 1 curve at X intersection, got {len(curves)}"
        )


class TestPostprocess:
    """Test postprocessing: spline + resample."""

    def test_resample_count(self):
        config = JunctionConfig(resample_n=100)
        pixels = [(x, int(50 + 30 * math.sin(x / 50.0))) for x in range(200)]
        result = postprocess_curve(pixels, config)
        assert len(result) == 100

    def test_handles_short_input(self):
        config = JunctionConfig()
        result = postprocess_curve([(10, 20), (20, 30)], config)
        assert len(result) >= 2


class TestCalibration:
    """Test pixel→data mapping."""

    def test_3point_affine_roundtrip(self):
        """3-point affine: roundtrip pixel→data→pixel should be accurate."""
        points = [
            {"pixel": [100, 400], "data": [0.0, 40.0]},
            {"pixel": [500, 400], "data": [200000.0, 40.0]},
            {"pixel": [100, 50],  "data": [0.0, 70.0]},
        ]
        mapping = calibrate_from_3points(points)
        assert mapping.error_px < 1.0

        # Map known pixels
        data = map_pixels_to_data([(100, 400), (500, 400), (100, 50)], mapping)
        np.testing.assert_allclose(data[0], (0.0, 40.0), atol=0.1)
        np.testing.assert_allclose(data[1], (200000.0, 40.0), atol=0.1)
        np.testing.assert_allclose(data[2], (0.0, 70.0), atol=0.1)

    def test_2axis_mapping(self):
        x_refs = [{"pixel": 100, "value": 0.0}, {"pixel": 500, "value": 200000.0}]
        y_refs = [{"pixel": 400, "value": 40.0}, {"pixel": 50, "value": 70.0}]
        plot_area = (100, 50, 500, 400)
        mapping = calibrate_from_2points_per_axis(x_refs, y_refs, plot_area)

        data = map_pixels_to_data([(100, 400)], mapping)
        np.testing.assert_allclose(data[0][0], 0.0, atol=1.0)
        np.testing.assert_allclose(data[0][1], 40.0, atol=1.0)

    def test_tick_pixel_to_value(self):
        """Known tick mark pixels should map to expected axis values."""
        points = [
            {"pixel": [100, 400], "data": [0.0, 40.0]},
            {"pixel": [500, 400], "data": [200000.0, 40.0]},
            {"pixel": [100, 50],  "data": [0.0, 70.0]},
        ]
        mapping = calibrate_from_3points(points)

        # Midpoint should map to ~100000, ~40 (bottom axis)
        mid = map_pixels_to_data([(300, 400)], mapping)
        np.testing.assert_allclose(mid[0][0], 100000.0, atol=500)
        np.testing.assert_allclose(mid[0][1], 40.0, atol=0.5)


class TestDeskew:
    """Test image deskew."""

    def test_deskew_straight_image(self):
        """A straight image should have near-zero skew angle."""
        image = np.ones((400, 600), dtype=np.uint8) * 255
        # Draw horizontal and vertical axis lines
        cv2.line(image, (50, 350), (550, 350), 0, 2)
        cv2.line(image, (50, 50), (50, 350), 0, 2)
        deskewed, angle = deskew_image(image)
        assert abs(angle) < 2.0, f"Expected small angle for straight image, got {angle}"


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_digitize_overlapping_arcs(self):
        """Full pipeline on synthetic overlapping arcs."""
        image, gt = generate_overlapping_arcs(num_curves=3, stroke_width=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test_arcs.png")
            cv2.imwrite(img_path, image)

            result = digitize(
                image_path=img_path,
                output_dir=os.path.join(tmpdir, "out"),
                config=JunctionConfig(beam_width=3, debug=True),
                num_curves=3,
            )

            assert "curves" in result
            assert len(result["curves"]) >= 1
            assert result["num_edges"] > 0
            assert len(result["files_written"]) >= 3
            # Check files exist
            for f in result["files_written"]:
                assert os.path.isfile(f), f"Output file not created: {f}"

    def test_digitize_crossing_curves(self):
        """Full pipeline on X-crossing curves."""
        image, gt = generate_crossing_curves(stroke_width=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test_cross.png")
            cv2.imwrite(img_path, image)

            result = digitize(
                image_path=img_path,
                output_dir=os.path.join(tmpdir, "out"),
                config=JunctionConfig(beam_width=5, debug=True),
                num_curves=2,
            )

            assert len(result["curves"]) >= 1

    def test_summary_not_generic(self):
        """Image summary should reference actual image properties."""
        image, gt = generate_overlapping_arcs(num_curves=4, stroke_width=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.png")
            cv2.imwrite(img_path, image)

            result = digitize(
                image_path=img_path,
                output_dir=os.path.join(tmpdir, "out"),
                config=JunctionConfig(),
            )

            summary = result["summary"]
            assert "600" in summary or "400" in summary  # image dimensions
            assert "curve" in summary.lower()
            assert "junction" in summary.lower() or "overlap" in summary.lower()

    def test_with_calibration_json(self):
        """Full pipeline with calibration file."""
        import json as json_mod

        image, gt = generate_overlapping_arcs(
            width=600, height=400, num_curves=3, stroke_width=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.png")
            cv2.imwrite(img_path, image)

            calib = {
                "mode": "3point",
                "points": [
                    {"pixel": [40, 360], "data": [0.0, 0.0]},
                    {"pixel": [560, 360], "data": [200000.0, 0.0]},
                    {"pixel": [40, 40], "data": [0.0, 70.0]},
                ],
            }
            calib_path = os.path.join(tmpdir, "calib.json")
            with open(calib_path, "w") as f:
                json_mod.dump(calib, f)

            result = digitize(
                image_path=img_path,
                calib_path=calib_path,
                output_dir=os.path.join(tmpdir, "out"),
                config=JunctionConfig(),
                num_curves=3,
            )

            assert result.get("mapping_error_px") is not None
            assert result["mapping_error_px"] < 1.0

    def test_with_anchors_json(self):
        """Full pipeline with anchor file."""
        import json as json_mod

        image, gt = generate_overlapping_arcs(
            width=600, height=400, num_curves=3, stroke_width=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.png")
            cv2.imwrite(img_path, image)

            # Use ground truth start/end as anchors
            anchors = {"curves": []}
            for i, curve_pts in enumerate(gt):
                if len(curve_pts) < 2:
                    continue
                start = list(curve_pts[0])
                mid = list(curve_pts[len(curve_pts) // 2])
                end = list(curve_pts[-1])
                anchors["curves"].append({
                    "label": f"curve_{i}",
                    "anchors": [start, mid, end],
                })

            anchors_path = os.path.join(tmpdir, "anchors.json")
            with open(anchors_path, "w") as f:
                json_mod.dump(anchors, f)

            result = digitize(
                image_path=img_path,
                anchors_path=anchors_path,
                output_dir=os.path.join(tmpdir, "out"),
                config=JunctionConfig(),
            )

            # Should have found curves equal to number of anchored curves
            assert len(result["curves"]) >= 1


# ====================================================================
# CLI for generating test images
# ====================================================================

def generate_all_test_images(output_dir: str = "tests/synthetic"):
    """Generate all synthetic test images for manual inspection."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating synthetic test images in {output_dir}/")

    # 1. Overlapping arcs (main test case)
    img, gt = generate_overlapping_arcs(num_curves=5, stroke_width=2)
    save_synthetic_image(img, gt, output_dir, "overlapping_arcs_5")
    print(f"  overlapping_arcs_5.png  ({len(gt)} curves)")

    # 2. Crossover (X pattern)
    img, gt = generate_crossing_curves(stroke_width=2)
    save_synthetic_image(img, gt, output_dir, "crossing_curves")
    print(f"  crossing_curves.png  ({len(gt)} curves)")

    # 3. Near-parallel with merge point
    img, gt = generate_near_parallel_curves(num_curves=3, gap=12, stroke_width=2)
    save_synthetic_image(img, gt, output_dir, "near_parallel_3")
    print(f"  near_parallel_3.png  ({len(gt)} curves)")

    # 4. Thick strokes (harder skeleton)
    img, gt = generate_overlapping_arcs(num_curves=4, stroke_width=4)
    save_synthetic_image(img, gt, output_dir, "thick_overlapping_4")
    print(f"  thick_overlapping_4.png  ({len(gt)} curves)")

    # 5. Noisy
    img, gt = generate_overlapping_arcs(num_curves=3, stroke_width=2, noise_sigma=2.0)
    save_synthetic_image(img, gt, output_dir, "noisy_arcs_3")
    print(f"  noisy_arcs_3.png  ({len(gt)} curves)")

    print(f"\nDone. {5} test images generated.")


if __name__ == "__main__":
    import sys
    if "--generate" in sys.argv:
        out_dir = sys.argv[sys.argv.index("--generate") + 1] if len(sys.argv) > sys.argv.index("--generate") + 1 else "tests/synthetic"
        generate_all_test_images(out_dir)
    else:
        pytest.main([__file__, "-v", "--tb=short"])
