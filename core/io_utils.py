"""
I/O utilities for curve digitization.

Handles reading / writing artifacts and building download zip archives.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional


def write_metrics_json(metrics_dict: Dict[str, Any], path: Path) -> None:
    """Write a metrics dict to *path* as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics_dict, f, indent=2, default=str)


def write_report_json(report_dict: Dict[str, Any], path: Path) -> None:
    """Write the full report dict to *path* as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report_dict, f, indent=2, default=str)


def build_download_zip(job_dir: Path, artifact_names: Optional[List[str]] = None) -> bytes:
    """Create an in-memory zip of all job artifacts.

    Parameters
    ----------
    job_dir : Path
        Directory containing the artifacts.
    artifact_names : list[str] | None
        Specific file names to include.  If None all files in *job_dir* are used.

    Returns
    -------
    bytes
        Zip archive content.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if artifact_names:
            for name in artifact_names:
                fpath = job_dir / name
                if fpath.exists():
                    zf.write(str(fpath), arcname=name)
        else:
            for fpath in sorted(job_dir.iterdir()):
                if fpath.is_file():
                    zf.write(str(fpath), arcname=fpath.name)
    return buf.getvalue()


def load_report_json(path: Path) -> Dict[str, Any]:
    """Load a report.json and return its contents."""
    with open(path, "r") as f:
        return json.load(f)


def list_job_artifacts(job_dir: Path) -> List[str]:
    """List file names inside a job output directory."""
    if not job_dir.is_dir():
        return []
    return sorted(f.name for f in job_dir.iterdir() if f.is_file())
