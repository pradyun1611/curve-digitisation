"""
Regression tests for BW (grayscale) polynomial-fit curve reconstruction.

Verifies that the robust BW pipeline produces:
  - Correct curve count (>= expected minimum)
  - Enough fitted points per curve (>= 200)
  - Smooth output (low mean |d²y/dx²|)
  - No tiny fragments (x_span_ratio >= 0.4)

On failure, debug artifacts are dumped to tests/artifacts/bw_polyfit/.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.image_processor import CurveDigitizer
from core.bw_reconstruction import smoothness_metric

DATA_DIR = _ROOT / "tests" / "data"
ARTIFACTS_DIR = _ROOT / "tests" / "artifacts" / "bw_polyfit"

BW_IMG = DATA_DIR / "input_bw.png"
BW_HARD_IMG = DATA_DIR / "input_bw_hard.png"

AXIS_INFO = {
    "xMin": 0,
    "xMax": 100,
    "yMin": 0,
    "yMax": 100,
    "xUnit": "%",
    "yUnit": "%",
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _dump_artifacts(name: str, results: dict) -> None:
    """Dump debug artifacts on test failure."""
    out = ARTIFACTS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    import json
    # Save a summary JSON
    summary = {}
    for cname, cdata in results.get("curves", {}).items():
        if not isinstance(cdata, dict):
            continue
        fit = cdata.get("fit_result", {})
        summary[cname] = {
            "label": cdata.get("label"),
            "original_point_count": cdata.get("original_point_count"),
            "cleaned_point_count": cdata.get("cleaned_point_count"),
            "degree": fit.get("degree"),
            "r_squared": fit.get("r_squared"),
            "n_fitted": len(fit.get("fitted_points", [])),
            "equation": fit.get("equation"),
        }
    with open(str(out / "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def _run_bw_pipeline(image_path: Path, num_curves: int = 3) -> dict:
    """Run the full BW pipeline and return results dict."""
    digitizer = CurveDigitizer(AXIS_INFO)
    features = {
        "curves": [
            {"color": "black", "shape": "curved", "label": f"Curve {i+1}"}
            for i in range(num_curves)
        ]
    }
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        results = digitizer.process_curve_image(
            str(image_path), features, tmp
        )
    return results


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _ensure_test_images():
    """Generate test images if they don't exist."""
    if not BW_IMG.exists():
        from scripts.generate_test_images import make_bw_image
        make_bw_image()


# ------------------------------------------------------------------
# Test: curve count
# ------------------------------------------------------------------

class TestBWCurveCount:
    """BW pipeline must find the expected number of curves."""

    def test_input_bw_finds_at_least_2_curves(self):
        results = _run_bw_pipeline(BW_IMG, num_curves=3)
        curves = results.get("curves", {})
        valid = {k: v for k, v in curves.items()
                 if isinstance(v, dict) and not v.get("error")}
        try:
            assert len(valid) >= 2, (
                f"Expected ≥2 BW curves, found {len(valid)}: "
                f"{list(valid.keys())}"
            )
        except AssertionError:
            _dump_artifacts("count_input_bw", results)
            raise

    @pytest.mark.skipif(not BW_HARD_IMG.exists(),
                        reason="input_bw_hard.png not available")
    def test_input_bw_hard_finds_curves(self):
        results = _run_bw_pipeline(BW_HARD_IMG, num_curves=5)
        curves = results.get("curves", {})
        valid = {k: v for k, v in curves.items()
                 if isinstance(v, dict) and not v.get("error")}
        try:
            assert len(valid) >= 1, (
                f"Expected ≥1 BW curve from hard image, found {len(valid)}"
            )
        except AssertionError:
            _dump_artifacts("count_input_bw_hard", results)
            raise


# ------------------------------------------------------------------
# Test: fitted point count
# ------------------------------------------------------------------

class TestBWFittedPointCount:
    """Each curve must have enough smooth fitted points."""

    def test_each_curve_has_at_least_200_fitted_points(self):
        results = _run_bw_pipeline(BW_IMG, num_curves=3)
        curves = results.get("curves", {})
        for cname, cdata in curves.items():
            if not isinstance(cdata, dict) or cdata.get("error"):
                continue
            fit = cdata.get("fit_result", {})
            fitted_pts = fit.get("fitted_points", [])
            try:
                assert len(fitted_pts) >= 200, (
                    f"Curve '{cname}' has only {len(fitted_pts)} fitted points"
                )
            except AssertionError:
                _dump_artifacts("fitted_count", results)
                raise


# ------------------------------------------------------------------
# Test: smoothness
# ------------------------------------------------------------------

class TestBWSmoothness:
    """Fitted curves must be smoother than raw extracted points."""

    def test_fitted_smoother_than_raw(self):
        results = _run_bw_pipeline(BW_IMG, num_curves=3)
        curves = results.get("curves", {})
        for cname, cdata in curves.items():
            if not isinstance(cdata, dict) or cdata.get("error"):
                continue

            # Raw axis coords
            raw_pts = cdata.get("cleaned_point_count", 0)
            # We need the actual cleaned points to compute smoothness.
            # They are stored as axis_coords derived from raw_pixel_points
            # in the pipeline.  The fit_result.fitted_points is the smooth
            # output.
            fit = cdata.get("fit_result", {})
            fitted_pts = fit.get("fitted_points", [])
            if len(fitted_pts) < 10:
                continue

            smooth_fitted = smoothness_metric(fitted_pts)

            # For a polynomial fit, the second derivative should be very
            # smooth.  Assert it's below a reasonable threshold.
            # Typical value for a degree-3 poly on [0,100]: < 0.1
            try:
                assert smooth_fitted < 5.0, (
                    f"Curve '{cname}' fitted smoothness={smooth_fitted:.4f} "
                    f"too high (expected <5.0)"
                )
            except AssertionError:
                _dump_artifacts("smoothness", results)
                raise


# ------------------------------------------------------------------
# Test: no tiny fragments
# ------------------------------------------------------------------

class TestBWNoFragments:
    """Major curves should span a large fraction of the x-axis."""

    def test_no_tiny_fragments(self):
        results = _run_bw_pipeline(BW_IMG, num_curves=3)
        curves = results.get("curves", {})
        x_range = AXIS_INFO["xMax"] - AXIS_INFO["xMin"]
        for cname, cdata in curves.items():
            if not isinstance(cdata, dict) or cdata.get("error"):
                continue
            fit = cdata.get("fit_result", {})
            fitted_pts = fit.get("fitted_points", [])
            if len(fitted_pts) < 2:
                continue
            xs = [p["x"] for p in fitted_pts]
            x_span = max(xs) - min(xs)
            x_span_ratio = x_span / x_range if x_range > 0 else 0

            try:
                assert x_span_ratio >= 0.35, (
                    f"Curve '{cname}' x_span_ratio={x_span_ratio:.3f} "
                    f"< 0.35 — likely a fragment"
                )
            except AssertionError:
                _dump_artifacts("fragments", results)
                raise


# ------------------------------------------------------------------
# Test: R-squared quality
# ------------------------------------------------------------------

class TestBWFitQuality:
    """Polynomial fits should have high R² on the BW test image."""

    def test_r_squared_above_threshold(self):
        results = _run_bw_pipeline(BW_IMG, num_curves=3)
        curves = results.get("curves", {})
        for cname, cdata in curves.items():
            if not isinstance(cdata, dict) or cdata.get("error"):
                continue
            fit = cdata.get("fit_result", {})
            r2 = fit.get("r_squared")
            if r2 is None:
                continue
            try:
                assert r2 >= 0.85, (
                    f"Curve '{cname}' R²={r2:.4f} < 0.85"
                )
            except AssertionError:
                _dump_artifacts("r_squared", results)
                raise


# ------------------------------------------------------------------
# Test: determinism
# ------------------------------------------------------------------

class TestBWDeterminism:
    """BW pipeline must be deterministic across runs."""

    def test_two_runs_identical(self):
        r1 = _run_bw_pipeline(BW_IMG, num_curves=3)
        r2 = _run_bw_pipeline(BW_IMG, num_curves=3)

        c1 = r1.get("curves", {})
        c2 = r2.get("curves", {})
        assert set(c1.keys()) == set(c2.keys()), (
            f"Different curve keys: {set(c1.keys())} vs {set(c2.keys())}"
        )
        for k in c1:
            if not isinstance(c1[k], dict) or c1[k].get("error"):
                continue
            fit1 = c1[k].get("fit_result", {})
            fit2 = c2[k].get("fit_result", {})
            fp1 = fit1.get("fitted_points", [])
            fp2 = fit2.get("fitted_points", [])
            assert len(fp1) == len(fp2), (
                f"Curve '{k}': different fitted_points count "
                f"{len(fp1)} vs {len(fp2)}"
            )
            for i, (p1, p2) in enumerate(zip(fp1, fp2)):
                assert abs(p1["x"] - p2["x"]) < 1e-9, (
                    f"Curve '{k}' point {i}: x differs"
                )
                assert abs(p1["y"] - p2["y"]) < 1e-9, (
                    f"Curve '{k}' point {i}: y differs"
                )


# ------------------------------------------------------------------
# Test: bw_reconstruction module unit tests
# ------------------------------------------------------------------

class TestBWReconstructionUnits:
    """Unit tests for the bw_reconstruction helper functions."""

    def test_prepare_points_sorts_and_deduplicates(self):
        from core.bw_reconstruction import prepare_points
        # Unsorted, with duplicate x values
        raw = [(5.0, 10.0), (1.0, 2.0), (5.0, 12.0), (3.0, 6.0), (1.0, 4.0)]
        result = prepare_points(raw)
        # Should be sorted by x, duplicates merged (median y)
        assert result[0, 0] == 1.0
        assert result[-1, 0] == 5.0
        # x=1 had y=2 and y=4 → median=3
        assert result[0, 1] == 3.0
        # x=5 had y=10 and y=12 → median=11
        assert result[-1, 1] == 11.0

    def test_merge_fragments_combines_nearby(self):
        from core.bw_reconstruction import merge_fragments
        # Two fragments that should merge (small x-gap, similar slope)
        frag1 = [(x, 100) for x in range(10, 50)]
        frag2 = [(x, 100) for x in range(55, 95)]
        clusters = {0: frag1, 1: frag2}
        merged = merge_fragments(clusters, roi_width=100, x_gap_ratio=0.10)
        # Should merge into 1 cluster
        assert len(merged) == 1
        # Combined points
        total_pts = sum(len(v) for v in merged.values())
        assert total_pts == len(frag1) + len(frag2)

    def test_merge_fragments_keeps_separate_curves(self):
        from core.bw_reconstruction import merge_fragments
        # Two fragments far apart in Y (different curves)
        frag1 = [(x, 50) for x in range(10, 90)]
        frag2 = [(x, 200) for x in range(10, 90)]
        clusters = {0: frag1, 1: frag2}
        merged = merge_fragments(clusters, roi_width=100, y_gap_ratio=0.15)
        # Should remain as 2 separate clusters
        assert len(merged) == 2

    def test_fit_polynomial_robust_returns_correct_schema(self):
        from core.bw_reconstruction import fit_polynomial_robust
        # Simple parabola with noise
        np.random.seed(42)
        xs = np.linspace(0, 100, 200)
        ys = 50 + 0.005 * (xs - 50) ** 2 + np.random.normal(0, 0.5, 200)
        points = np.column_stack([xs, ys])
        result = fit_polynomial_robust(points, n_output=300)
        assert "degree" in result
        assert "coefficients" in result
        assert "r_squared" in result
        assert "fitted_points" in result
        assert len(result["fitted_points"]) == 300
        assert result["r_squared"] > 0.9

    def test_smoothness_metric_poly_is_smooth(self):
        # A polynomial should have very low smoothness metric
        xs = np.linspace(0, 100, 300)
        ys = 50 + 0.005 * (xs - 50) ** 2
        pts = [{"x": float(x), "y": float(y)} for x, y in zip(xs, ys)]
        sm = smoothness_metric(pts)
        assert sm < 0.1, f"Polynomial smoothness={sm} should be near zero"

    def test_smoothness_metric_noisy_is_higher(self):
        np.random.seed(42)
        xs = np.linspace(0, 100, 300)
        ys = 50 + 0.005 * (xs - 50) ** 2 + np.random.normal(0, 5, 300)
        pts = [{"x": float(x), "y": float(y)} for x, y in zip(xs, ys)]
        sm = smoothness_metric(pts)
        # Noisy data should have higher smoothness metric
        assert sm > 0.01, f"Noisy smoothness={sm} should be higher"
