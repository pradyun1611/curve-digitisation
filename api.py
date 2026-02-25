"""
FastAPI backend for Curve Digitization.

Endpoints
---------
POST /api/analyze       – Upload image + query → run pipeline, return results + metrics.
GET  /api/jobs/{job_id} – Retrieve job report (metrics summary, artifacts list).
GET  /api/jobs/{job_id}/download – Download a zip of all artifacts.
POST /api/evaluate      – (Optional) Upload ground truth for a job and recompute metrics.
GET  /health            – Health-check.

Run with:
    uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# SSL fix ----------------------------------------------------------------
try:
    import local_config  # noqa: F401
except ImportError:
    pass

load_dotenv()

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import JSONResponse, Response, StreamingResponse
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "fastapi is not installed. Run: pip install 'fastapi[standard]' uvicorn"
    ) from _exc

from core.io_utils import build_download_zip, list_job_artifacts, load_report_json
from core.metrics import (
    compute_ground_truth_metrics,
    compute_self_consistency_metrics,
    parse_ground_truth_csv,
    parse_ground_truth_json,
)
from core.openai_client import OpenAIClient
from core.pipeline import run_pipeline
from core.reconstruction import build_masks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")

app = FastAPI(title="Curve Digitization API", version="1.0.0")


def _numpy_safe(obj: Any) -> Any:
    """Convert numpy types → native Python for JSON serialization."""
    import numpy as _np
    if isinstance(obj, _np.integer):
        return int(obj)
    if isinstance(obj, _np.floating):
        return float(obj)
    if isinstance(obj, _np.bool_):
        return bool(obj)
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    return str(obj)


@app.exception_handler(Exception)
async def _global_exception_handler(request: Any, exc: Exception) -> JSONResponse:
    """Return a structured JSON error for any unhandled exception."""
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__},
    )


def _get_openai_client() -> OpenAIClient:
    """Create an OpenAI client from env vars."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    endpoint = os.getenv("AZURE_ENDPOINT", "")
    deployment = os.getenv("AZURE_DEPLOYMENT_NAME", "")
    if not all([api_key, endpoint, deployment]):
        raise HTTPException(500, "Azure OpenAI credentials not configured in environment.")
    return OpenAIClient(api_key, endpoint, deployment)


# ------------------------------------------------------------------
# POST /api/analyze
# ------------------------------------------------------------------
@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    query: str = Form("Extract curves from this performance chart image"),
    ground_truth: Optional[UploadFile] = File(None),
    ground_truth_format: str = Form("csv"),
):
    """Upload an image, run full pipeline and return results with metrics."""
    job_id = uuid.uuid4().hex[:12]
    logger.info("[%s] /api/analyze: received image=%s query=%s", job_id, image.filename, query[:60])

    client = _get_openai_client()

    # Save uploaded image to temp
    tmp_dir = Path(tempfile.mkdtemp())
    img_path = tmp_dir / (image.filename or "upload.png")
    img_bytes = await image.read()
    img_path.write_bytes(img_bytes)

    # Encode
    import base64
    image_base64 = base64.b64encode(img_bytes).decode("utf-8")

    # Azure OpenAI calls
    axis_info = client.extract_axis_info(image_base64, query)
    features = client.extract_curve_features(image_base64)

    # Read optional ground truth
    gt_text: Optional[str] = None
    if ground_truth:
        gt_text = (await ground_truth.read()).decode("utf-8")

    # Run pipeline
    result = run_pipeline(
        str(img_path),
        axis_info,
        features,
        OUTPUT_DIR,
        job_id=job_id,
        ground_truth_text=gt_text,
        ground_truth_format=ground_truth_format,
    )

    # Cleanup temp
    shutil.rmtree(str(tmp_dir), ignore_errors=True)

    # Return as JSON with numpy-safe serialization
    payload = json.loads(json.dumps(result.to_dict(), default=_numpy_safe))
    return payload


# ------------------------------------------------------------------
# GET /api/jobs/{job_id}
# ------------------------------------------------------------------
@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Retrieve the report for a completed job."""
    report_path = Path(OUTPUT_DIR) / job_id / "report.json"
    if not report_path.exists():
        raise HTTPException(404, f"Job {job_id} not found.")
    return load_report_json(report_path)


# ------------------------------------------------------------------
# GET /api/jobs/{job_id}/download
# ------------------------------------------------------------------
@app.get("/api/jobs/{job_id}/download")
async def download_job(job_id: str):
    """Download a zip archive of all artifacts for a job."""
    job_dir = Path(OUTPUT_DIR) / job_id
    if not job_dir.is_dir():
        raise HTTPException(404, f"Job {job_id} not found.")

    artifacts = list_job_artifacts(job_dir)
    zip_bytes = build_download_zip(job_dir, artifacts)

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={job_id}_artifacts.zip"},
    )


# ------------------------------------------------------------------
# POST /api/evaluate
# ------------------------------------------------------------------
@app.post("/api/evaluate")
async def evaluate(
    job_id: str = Form(...),
    ground_truth: UploadFile = File(...),
    ground_truth_format: str = Form("csv"),
):
    """Upload ground truth for an existing job and recompute metrics."""
    job_dir = Path(OUTPUT_DIR) / job_id
    report_path = job_dir / "report.json"
    if not report_path.exists():
        raise HTTPException(404, f"Job {job_id} not found.")

    report = load_report_json(report_path)
    gt_text = (await ground_truth.read()).decode("utf-8")

    # Parse ground truth
    if ground_truth_format == "json":
        gt_series = parse_ground_truth_json(gt_text)
    else:
        gt_series = parse_ground_truth_csv(gt_text)

    # Build extracted series from report
    ext_series: dict = {}
    for k, v in report.get("curves", {}).items():
        ac = v.get("axis_coords") or []
        if ac:
            ext_series[k] = [(p[0], p[1]) for p in ac]
        fp = (v.get("fit_result") or {}).get("fitted_points", [])
        if fp and k not in ext_series:
            ext_series[k] = [(p["x"], p["y"]) for p in fp]

    # Re-use pixel metrics from existing report if available
    from core.types import MetricsResult
    existing = report.get("metrics")
    pixel_metrics = MetricsResult.from_dict(existing) if existing else None

    metrics = compute_ground_truth_metrics(
        ext_series, gt_series, pixel_metrics=pixel_metrics, job_id=job_id,
    )

    # Update report
    report["metrics"] = metrics.to_dict()
    report["metrics_summary"] = {
        "mode": metrics.mode,
        "delta_value": metrics.delta_value,
        "delta_norm": metrics.delta_norm,
        "iou": metrics.iou,
        "ssim": metrics.ssim,
        "delta_value_data": metrics.delta_value_data,
        "rmse_y": metrics.rmse_y,
        "mapping_status": metrics.mapping_status,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Also update metrics.json
    with open(job_dir / "metrics.json", "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)

    return metrics.to_dict()


# ------------------------------------------------------------------
# GET /health
# ------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}
