"""
Smoke tests for the full pipeline.

Uses a synthetic solid-color block image so no Azure credentials are needed.
Tests that the pipeline produces the expected artifacts on disk.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Fixture: create a synthetic test image with a horizontal red line
# ---------------------------------------------------------------------------
@pytest.fixture()
def synthetic_image(tmp_path: Path) -> Path:
    """Create a 200x150 white image with a red horizontal stripe."""
    img = PILImage.new("RGB", (200, 150), (255, 255, 255))
    pixels = img.load()
    for x in range(20, 180):
        for y in range(70, 80):
            pixels[x, y] = (255, 0, 0)  # red stripe
    path = tmp_path / "test_curve.png"
    img.save(str(path))
    return path


# ---------------------------------------------------------------------------
# Helpers – build minimal axis_info & features dicts
# ---------------------------------------------------------------------------
AXIS_INFO = {
    "xMin": 0, "xMax": 100,
    "yMin": 0, "yMax": 100,
    "xUnit": "%", "yUnit": "%",
    "imageDescription": "test",
}

FEATURES = {
    "curves": [
        {"color": "red", "shape": "curved", "label": "Test", "trend": "flat"}
    ],
    "numerical_data_visible": False,
    "grid_present": False,
}


class TestPipelineSmoke:
    """Smoke-test the pipeline with a synthetic image (no API calls)."""

    def test_artifacts_created(self, synthetic_image: Path, tmp_path: Path):
        """Pipeline should create metrics.json, report.json, debug.json, and images."""
        from core.pipeline import run_pipeline

        output_dir = tmp_path / "output"
        result = run_pipeline(
            str(synthetic_image),
            AXIS_INFO,
            FEATURES,
            str(output_dir),
            job_id="smoke_test",
        )

        job_dir = output_dir / "smoke_test"
        assert job_dir.is_dir()

        # Check expected artifacts
        assert (job_dir / "metrics.json").exists()
        assert (job_dir / "report.json").exists()
        assert (job_dir / "debug.json").exists()
        assert (job_dir / "reconstructed_plot.png").exists()
        assert (job_dir / "overlay_comparison.png").exists()
        # Debug mask artifacts
        assert (job_dir / "original_series_mask.png").exists()
        assert (job_dir / "reconstructed_mask.png").exists()
        assert (job_dir / "mask_diff.png").exists()

        # Validate metrics.json is valid JSON
        with open(job_dir / "metrics.json") as f:
            metrics = json.load(f)
        assert "delta_value" in metrics
        assert isinstance(metrics["delta_value"], (int, float))
        # delta_value should be finite for a synthetic plot with detected series
        # (could be NaN if masks are empty, but our red stripe should be detected)
        assert "ssim" in metrics  # SSIM key must be present
        assert "iou" in metrics
        assert "precision" in metrics
        assert "recall" in metrics

        # Validate report.json
        with open(job_dir / "report.json") as f:
            report = json.load(f)
        assert report["job_id"] == "smoke_test"
        assert "curves" in report
        assert "axis_info" in report

        # Validate debug.json has mapping info
        with open(job_dir / "debug.json") as f:
            debug = json.load(f)
        assert "has_mapping" in debug
        assert "plot_area_width" in debug
        assert "plot_area_height" in debug
        # When mapping is available, check round-trip error fields
        if debug.get("has_mapping"):
            assert "mapping_roundtrip_error_mean_px" in debug
            assert "mapping_roundtrip_error_p95_px" in debug

    def test_result_object(self, synthetic_image: Path, tmp_path: Path):
        """ExtractionResult should have the right shape."""
        from core.pipeline import run_pipeline

        result = run_pipeline(
            str(synthetic_image),
            AXIS_INFO,
            FEATURES,
            str(tmp_path / "out2"),
            job_id="smoke2",
        )

        assert result.job_id == "smoke2"
        assert result.metrics is not None
        assert result.metrics.delta_value >= 0
        assert result.metrics.mode == "self_consistency"
        assert len(result.artifacts) >= 5  # metrics, report, debug, 2 images + masks
        assert result.debug  # debug dict should be populated

        # delta_value should be > 0 when reconstruction uses fitted polynomial
        # (different from raw extracted pixels)
        # Note: for a simple synthetic stripe this may be small but non-zero
        # unless the fit perfectly matches the extraction

        # Per-series regression quality should be in debug info
        if "series_regression" in result.debug:
            for sname, reg in result.debug["series_regression"].items():
                assert "r2_score" in reg
                assert "pearson_r" in reg
                assert "n_points" in reg

    def test_ground_truth_csv(self, synthetic_image: Path, tmp_path: Path):
        """Pipeline should handle optional ground-truth CSV."""
        from core.pipeline import run_pipeline

        gt_csv = "x,y\n0,50\n50,50\n100,50\n"
        result = run_pipeline(
            str(synthetic_image),
            AXIS_INFO,
            FEATURES,
            str(tmp_path / "out3"),
            job_id="smoke_gt",
            ground_truth_text=gt_csv,
            ground_truth_format="csv",
        )

        assert result.metrics is not None
        # If ground truth evaluation ran, mode should be ground_truth
        assert result.metrics.mode == "ground_truth"

    def test_to_dict_roundtrip(self, synthetic_image: Path, tmp_path: Path):
        """to_dict() should produce a JSON-serializable dict."""
        from core.pipeline import run_pipeline

        result = run_pipeline(
            str(synthetic_image),
            AXIS_INFO,
            FEATURES,
            str(tmp_path / "out4"),
            job_id="roundtrip",
        )

        d = result.to_dict()
        # Should be JSON-serializable
        serialized = json.dumps(d, default=str)
        assert isinstance(serialized, str)
        assert json.loads(serialized)["job_id"] == "roundtrip"
