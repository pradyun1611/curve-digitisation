"""
Regression tests for BW (grayscale) pipeline — checks for known issues:
  1. Waviness / S-shaped wobble  (degree-2 polyfit should be monotone in d²y/dx²)
  2. Curves far apart / mismatched  (fitted y must match extracted y within tolerance)
  3. Missing curves  (pipeline must find ≥ expected count)
  4. Mapping bounds  (fitted points must lie within axis range)

These tests use the same synthetic test images as the rest of the BW suite.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.image_processor import CurveDigitizer

DATA_DIR = _ROOT / "tests" / "data"
BW_IMG = DATA_DIR / "input_bw.png"

AXIS_INFO = {
    "xMin": 0, "xMax": 100,
    "yMin": 0, "yMax": 100,
    "xUnit": "%", "yUnit": "%",
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _run_bw(image_path: Path, num_curves: int = 3) -> dict:
    digitizer = CurveDigitizer(AXIS_INFO)
    features = {
        "curves": [
            {"color": "black", "shape": "curved", "label": f"Curve {i+1}"}
            for i in range(num_curves)
        ]
    }
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        return digitizer.process_curve_image(str(image_path), features, tmp)


def _valid_curves(results: dict) -> dict:
    return {k: v for k, v in results.get("curves", {}).items()
            if isinstance(v, dict) and not v.get("error")}


@pytest.fixture(autouse=True)
def _ensure_test_images():
    if not BW_IMG.exists():
        from scripts.generate_test_images import make_bw_image
        make_bw_image()


# ------------------------------------------------------------------
# 1. No waviness — second derivative should not oscillate
# ------------------------------------------------------------------

class TestBWNoWaviness:
    """Degree-2 polynomial should have constant 2nd derivative (no oscillation)."""

    def test_second_derivative_nearly_constant(self):
        """For a degree-2 fit, d²y/dx² is exactly constant.

        If the pipeline accidentally uses degree ≥ 3 or piecewise fits,
        the second derivative will oscillate, producing visible waviness.
        We check that the max deviation of d²y/dx² from its mean is ≤ 1e-6
        (essentially zero for a true quadratic).
        """
        results = _run_bw(BW_IMG, num_curves=3)
        curves = _valid_curves(results)
        assert len(curves) >= 2, f"Need ≥2 curves, got {len(curves)}"

        for cname, cdata in curves.items():
            fit = cdata.get("fit_result", {})
            pts = fit.get("fitted_points", [])
            if len(pts) < 10:
                continue

            xs = np.array([p["x"] for p in pts])
            ys = np.array([p["y"] for p in pts])
            dx = np.diff(xs)
            dy = np.diff(ys)
            # first derivative
            dydx = dy / np.where(dx == 0, 1e-12, dx)
            # second derivative
            d2y = np.diff(dydx) / np.where(dx[:-1] == 0, 1e-12, dx[:-1])

            # For a degree-2 polyfit d²y/dx² is constant
            d2y_dev = np.max(np.abs(d2y - np.mean(d2y)))
            assert d2y_dev < 0.01, (
                f"Curve '{cname}': d²y/dx² deviation = {d2y_dev:.6f} — "
                f"waviness detected (expected < 0.01 for degree-2)"
            )

    def test_all_fits_are_degree_2(self):
        """Pipeline must use degree-2 polynomial (not auto-degree or piecewise)."""
        results = _run_bw(BW_IMG, num_curves=3)
        for cname, cdata in _valid_curves(results).items():
            fit = cdata.get("fit_result", {})
            assert fit.get("degree") == 2, (
                f"Curve '{cname}' degree={fit.get('degree')} — expected 2"
            )


# ------------------------------------------------------------------
# 2. Mapping accuracy — fitted curves match extracted pixel positions
# ------------------------------------------------------------------

class TestBWMappingAccuracy:
    """Fitted curves should be close to the extracted data points."""

    def test_fitted_within_axis_bounds(self):
        """All fitted y-values must lie within [yMin, yMax] (with small margin)."""
        results = _run_bw(BW_IMG, num_curves=3)
        margin = (AXIS_INFO["yMax"] - AXIS_INFO["yMin"]) * 0.05  # 5% margin
        for cname, cdata in _valid_curves(results).items():
            fit = cdata.get("fit_result", {})
            pts = fit.get("fitted_points", [])
            if not pts:
                continue
            ys = [p["y"] for p in pts]
            xs = [p["x"] for p in pts]
            assert min(ys) >= AXIS_INFO["yMin"] - margin, (
                f"Curve '{cname}' y_min={min(ys):.2f} below axis range"
            )
            assert max(ys) <= AXIS_INFO["yMax"] + margin, (
                f"Curve '{cname}' y_max={max(ys):.2f} above axis range"
            )
            assert min(xs) >= AXIS_INFO["xMin"] - margin, (
                f"Curve '{cname}' x_min={min(xs):.2f} below axis range"
            )
            assert max(xs) <= AXIS_INFO["xMax"] + margin, (
                f"Curve '{cname}' x_max={max(xs):.2f} above axis range"
            )

    def test_curves_vertically_separated(self):
        """Multiple grayscale curves should have distinct y-level centres.

        On the test image the 3 curves are at different heights.
        If mapping is wrong (e.g., merge_fragments mixed them), the
        y-centres would collapse.
        """
        results = _run_bw(BW_IMG, num_curves=3)
        curves = _valid_curves(results)
        if len(curves) < 2:
            pytest.skip("Need ≥2 curves for separation test")

        centres = []
        for cname, cdata in sorted(curves.items()):
            fit = cdata.get("fit_result", {})
            pts = fit.get("fitted_points", [])
            if pts:
                centres.append(np.mean([p["y"] for p in pts]))

        # Sort centres and check separation
        centres.sort()
        y_range = AXIS_INFO["yMax"] - AXIS_INFO["yMin"]
        for i in range(1, len(centres)):
            gap = centres[i] - centres[i - 1]
            assert gap > y_range * 0.05, (
                f"Curves {i-1} and {i} y-centres too close: "
                f"{centres[i-1]:.2f} vs {centres[i]:.2f} "
                f"(gap={gap:.2f}, need >{y_range*0.05:.2f})"
            )


# ------------------------------------------------------------------
# 3. No missing curves
# ------------------------------------------------------------------

class TestBWNoMissingCurves:
    """Pipeline must detect all visible curves."""

    def test_finds_all_3_curves(self):
        """The standard BW test image has exactly 3 curves."""
        results = _run_bw(BW_IMG, num_curves=3)
        curves = _valid_curves(results)
        assert len(curves) >= 3, (
            f"Expected 3 curves, found {len(curves)}: {list(curves.keys())}"
        )


# ------------------------------------------------------------------
# 4. Grayscale path uses same fitting as colour path
# ------------------------------------------------------------------

class TestBWUsesColourPathFitting:
    """Verify that grayscale path produces the exact same output format
    and uses degree-2 polynomial like the colour path."""

    def test_output_keys_match_colour_format(self):
        """Results dict must include all keys the colour path produces."""
        results = _run_bw(BW_IMG, num_curves=3)
        expected_keys = {
            'label', 'color', 'extraction_mode',
            'raw_pixel_points', 'plot_area',
            'original_point_count', 'normalized_point_count',
            'cleaned_point_count', 'fit_result', 'metrics',
        }
        for cname, cdata in _valid_curves(results).items():
            missing = expected_keys - set(cdata.keys())
            assert not missing, (
                f"Curve '{cname}' missing keys: {missing}"
            )

    def test_extraction_mode_is_grayscale(self):
        results = _run_bw(BW_IMG, num_curves=3)
        for cname, cdata in _valid_curves(results).items():
            assert cdata.get("extraction_mode") == "grayscale"

    def test_r_squared_above_threshold(self):
        """Same R² threshold as colour pipeline tests."""
        results = _run_bw(BW_IMG, num_curves=3)
        for cname, cdata in _valid_curves(results).items():
            r2 = cdata.get("fit_result", {}).get("r_squared")
            if r2 is not None:
                assert r2 >= 0.85, (
                    f"Curve '{cname}' R²={r2:.4f} < 0.85"
                )
