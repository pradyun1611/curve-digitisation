"""
Unit tests for core.metrics module.

These tests use synthetic binary masks to validate IoU, SSIM, precision,
recall, and delta_value calculations without needing real images.
"""

import numpy as np
import pytest

from core.metrics import compute_self_consistency_metrics, compute_series_regression


class TestSelfConsistencyMetrics:
    """Tests for compute_self_consistency_metrics."""

    def test_identical_masks(self):
        """Identical masks should give perfect scores."""
        mask = np.zeros((100, 200), dtype=bool)
        mask[40:60, 20:180] = True  # horizontal band

        m = compute_self_consistency_metrics(mask, mask.copy(), 200, 100)

        assert m.iou == pytest.approx(1.0, abs=1e-6)
        assert m.precision == pytest.approx(1.0, abs=1e-6)
        assert m.recall == pytest.approx(1.0, abs=1e-6)
        assert m.delta_value == pytest.approx(0.0, abs=1e-6)
        assert m.delta_norm == pytest.approx(0.0, abs=1e-6)

    def test_disjoint_masks(self):
        """Completely non-overlapping masks should give IoU = 0."""
        a = np.zeros((100, 200), dtype=bool)
        b = np.zeros((100, 200), dtype=bool)
        a[10:20, :] = True
        b[80:90, :] = True

        m = compute_self_consistency_metrics(a, b, 200, 100)

        assert m.iou == pytest.approx(0.0, abs=1e-6)
        assert m.precision == pytest.approx(0.0, abs=1e-6)
        assert m.recall == pytest.approx(0.0, abs=1e-6)
        assert m.delta_value > 0

    def test_partial_overlap(self):
        """Partially overlapping masks give 0 < IoU < 1."""
        a = np.zeros((100, 200), dtype=bool)
        b = np.zeros((100, 200), dtype=bool)
        a[30:60, 50:150] = True
        b[40:70, 50:150] = True  # shifted down by 10

        m = compute_self_consistency_metrics(a, b, 200, 100)

        assert 0.0 < m.iou < 1.0
        assert 0.0 < m.precision < 1.0
        assert 0.0 < m.recall < 1.0
        assert m.delta_value > 0

    def test_empty_masks(self):
        """All-zero masks should not crash and return NaN delta."""
        a = np.zeros((100, 200), dtype=bool)
        b = np.zeros((100, 200), dtype=bool)

        m = compute_self_consistency_metrics(a, b, 200, 100)

        # IoU of two empty sets is undefined – implementation returns 0
        assert m.iou == pytest.approx(0.0, abs=1e-6)
        # Empty masks produce NaN delta as a warning signal
        assert np.isnan(m.delta_value)

    def test_mapping_status_propagated(self):
        """mapping_status should be stored in the result."""
        mask = np.ones((50, 50), dtype=bool)
        m = compute_self_consistency_metrics(mask, mask, 50, 50, mapping_status="mapped")
        assert m.mapping_status == "mapped"

    def test_ssim_returned(self):
        """SSIM should be computed as a float (not None / N/A)."""
        a = np.zeros((100, 200), dtype=bool)
        b = np.zeros((100, 200), dtype=bool)
        a[20:80, 20:180] = True
        b[20:80, 20:180] = True

        m = compute_self_consistency_metrics(a, b, 200, 100)
        assert m.ssim is not None, "SSIM should not be None for reasonable-sized masks"
        assert isinstance(m.ssim, float)
        assert -1.0 <= m.ssim <= 1.0

    def test_ssim_identical_masks_near_one(self):
        """SSIM of identical masks should be close to 1."""
        mask = np.zeros((100, 200), dtype=bool)
        mask[30:70, 30:170] = True
        m = compute_self_consistency_metrics(mask, mask.copy(), 200, 100)
        assert m.ssim is not None
        assert m.ssim > 0.99

    def test_shifted_masks_delta_positive(self):
        """Shifted masks should produce delta > 0 (symmetric chamfer)."""
        a = np.zeros((100, 200), dtype=bool)
        b = np.zeros((100, 200), dtype=bool)
        a[40:45, 20:180] = True   # thin strip
        b[50:55, 20:180] = True   # shifted 10px down

        m = compute_self_consistency_metrics(a, b, 200, 100)
        assert m.delta_value > 5  # should be ~10px shift
        assert m.delta_pixels_p95 > 5
        assert np.isfinite(m.delta_value)


class TestDeltaNorm:
    """Tests for delta_norm correctness."""

    def test_delta_norm_range(self):
        """delta_norm should be in [0, ~1] for reasonable inputs."""
        a = np.zeros((100, 200), dtype=bool)
        b = np.zeros((100, 200), dtype=bool)
        a[40:60, :] = True
        b[45:65, :] = True

        m = compute_self_consistency_metrics(a, b, 200, 100)

        assert 0.0 <= m.delta_norm <= 1.0
        # delta_norm is delta_value / max(width, height)
        assert m.delta_norm == pytest.approx(
            m.delta_value / max(200, 100), abs=1e-6
        )


class TestSeriesRegression:
    """Tests for per-series regression quality metrics."""

    def test_perfect_line(self):
        """A perfect linear series should give R² ≈ 1.0 and |Pearson R| ≈ 1.0."""
        pts = [(float(x), 2.0 * x + 5.0) for x in range(100)]
        reg = compute_series_regression(pts, series_name="perfect_line")
        assert reg["r2_score"] == pytest.approx(1.0, abs=1e-6)
        assert abs(reg["pearson_r"]) == pytest.approx(1.0, abs=1e-6)
        assert reg["n_points"] == 100

    def test_near_linear_with_noise(self):
        """A near-linear series with small noise should give R² > 0.95."""
        rng = np.random.default_rng(42)
        pts = [(float(x), 3.0 * x + 1.0 + rng.normal(0, 0.5)) for x in range(200)]
        reg = compute_series_regression(pts, series_name="noisy_line")
        assert reg["r2_score"] > 0.95
        assert abs(reg["pearson_r"]) > 0.95

    def test_too_few_points(self):
        """Fewer than 3 points should return zeros gracefully."""
        reg = compute_series_regression([(1.0, 2.0), (3.0, 4.0)])
        assert reg["r2_score"] == 0.0
        assert reg["n_points"] == 0

    def test_empty_input(self):
        """Empty input should return zeros."""
        reg = compute_series_regression([])
        assert reg["r2_score"] == 0.0
        assert reg["pearson_r"] == 0.0
