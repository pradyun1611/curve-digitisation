"""
Type definitions for curve digitization pipeline.

Provides dataclasses for structured results, metrics, and reports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MappingResult:
    """Affine mapping between pixel space and data space.

    The mapping is defined in plot-area-local coordinates:
    *  pixel (0,0) = top-left of the plot-area crop
    *  data  = axis_info min/max
    """
    # 2x3 affine: [data_x, data_y] = M @ [px, py, 1]
    pixel_to_data_matrix: List[List[float]] = field(default_factory=list)
    # Inverse 2x3: [px, py] = M_inv @ [data_x, data_y, 1]
    data_to_pixel_matrix: List[List[float]] = field(default_factory=list)

    frame: str = "plot_area"       # always "Plot_area"
    x_direction: int = 1           # +1 = x increases rightward
    y_direction: int = 1           # +1 = y increases upward (data)

    plot_area_width: int = 0
    plot_area_height: int = 0

    # Round-trip quality
    mapping_roundtrip_error_mean_px: float = 0.0
    mapping_roundtrip_error_p95_px: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MappingResult":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AxisInfo:
    """Axis boundary and unit information extracted from a chart image."""
    xMin: Optional[float] = None
    xMax: Optional[float] = None
    yMin: Optional[float] = None
    yMax: Optional[float] = None
    xUnit: str = "units"
    yUnit: str = "units"
    imageDescription: str = ""

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AxisInfo":
        return cls(
            xMin=d.get("xMin"),
            xMax=d.get("xMax"),
            yMin=d.get("yMin"),
            yMax=d.get("yMax"),
            xUnit=d.get("xUnit", "units"),
            yUnit=d.get("yUnit", "units"),
            imageDescription=d.get("imageDescription", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def has_mapping(self) -> bool:
        """Return True when all four axis bounds are known numbers."""
        return all(v is not None for v in (self.xMin, self.xMax, self.yMin, self.yMax))


@dataclass
class CurveFeature:
    """A single curve/line detected in the image."""
    color: str = "unknown"
    shape: str = ""
    label: str = ""
    trend: str = ""
    approximate_values: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CurveFeature":
        return cls(
            color=d.get("color", "unknown"),
            shape=d.get("shape", ""),
            label=d.get("label", ""),
            trend=d.get("trend", ""),
            approximate_values=d.get("approximate_values", ""),
        )


@dataclass
class FitResult:
    """Result of polynomial curve fitting."""
    degree: int = 2
    coefficients: Optional[List[float]] = None
    r_squared: float = 0.0
    fitted_points: Optional[List[Dict[str, float]]] = None
    original_point_count: int = 0
    equation: str = ""
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FitResult":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CurveResult:
    """Processing result for a single curve."""
    label: str = ""
    color: str = ""
    original_point_count: int = 0
    normalized_point_count: int = 0
    cleaned_point_count: int = 0
    fit_result: Optional[FitResult] = None
    pixel_coords: Optional[List[List[float]]] = None
    raw_pixel_points: Optional[List[List[float]]] = None
    axis_coords: Optional[List[List[float]]] = None
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CurveResult":
        fit = d.get("fit_result")
        return cls(
            label=d.get("label", ""),
            color=d.get("color", ""),
            original_point_count=d.get("original_point_count", 0),
            normalized_point_count=d.get("normalized_point_count", 0),
            cleaned_point_count=d.get("cleaned_point_count", 0),
            fit_result=FitResult.from_dict(fit) if isinstance(fit, dict) else None,
            pixel_coords=d.get("pixel_coords"),
            raw_pixel_points=d.get("raw_pixel_points"),
            axis_coords=d.get("axis_coords"),
            error=d.get("error"),
        )


@dataclass
class MetricsResult:
    """Quality / deviation metrics produced by the evaluation system."""
    mode: str = "self_consistency"  # "self_consistency" | "ground_truth"

    # Primary fixed scalar (pixels)
    delta_value: float = 0.0
    delta_pixels_mean: float = 0.0
    delta_pixels_p95: float = 0.0
    delta_norm: float = 0.0

    # Mask-level quality
    iou: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    ssim: Optional[float] = None

    # Ground-truth data-space metrics (only when GT supplied)
    rmse_y: Optional[float] = None
    mae_y: Optional[float] = None
    max_abs_y: Optional[float] = None
    delta_value_data: Optional[float] = None

    # Context
    mapping_status: str = "pixel_only"  # "mapped" | "pixel_only"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Drop None ground-truth fields when not applicable
        if self.mode == "self_consistency":
            for k in ("rmse_y", "mae_y", "max_abs_y", "delta_value_data"):
                d.pop(k, None)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MetricsResult":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ExtractionResult:
    """Full result of the curve digitization pipeline."""
    job_id: str = ""
    image_path: str = ""
    image_dimensions: Dict[str, int] = field(default_factory=dict)
    axis_info: Optional[AxisInfo] = None
    curves: Dict[str, CurveResult] = field(default_factory=dict)
    metrics: Optional[MetricsResult] = None
    mapping: Optional[MappingResult] = None
    debug: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "job_id": self.job_id,
            "image_path": self.image_path,
            "image_dimensions": self.image_dimensions,
            "axis_info": self.axis_info.to_dict() if self.axis_info else {},
            "curves": {k: asdict(v) for k, v in self.curves.items()},
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "metrics_summary": self._metrics_summary(),
            "mapping": self.mapping.to_dict() if self.mapping else None,
            "debug": self.debug,
            "artifacts": self.artifacts,
            "timestamp": self.timestamp or datetime.now().isoformat(),
        }
        return d

    def _metrics_summary(self) -> Optional[Dict[str, Any]]:
        if self.metrics is None:
            return None
        m = self.metrics
        return {
            "mode": m.mode,
            "delta_value": m.delta_value,
            "delta_norm": m.delta_norm,
            "iou": m.iou,
            "ssim": m.ssim,
            "mapping_status": m.mapping_status,
        }

    @classmethod
    def from_legacy_dict(cls, d: Dict[str, Any], job_id: str = "") -> "ExtractionResult":
        """Convert old-format results dict to ExtractionResult."""
        axis_raw = d.get("axis_info", {})
        axis = AxisInfo.from_dict(axis_raw) if isinstance(axis_raw, dict) else None
        curves: Dict[str, CurveResult] = {}
        for k, v in d.get("curves", {}).items():
            curves[k] = CurveResult.from_dict(v) if isinstance(v, dict) else CurveResult(color=k)
        return cls(
            job_id=job_id,
            image_path=d.get("image_path", ""),
            image_dimensions=d.get("image_dimensions", {}),
            axis_info=axis,
            curves=curves,
            timestamp=d.get("timestamp", datetime.now().isoformat()),
        )
