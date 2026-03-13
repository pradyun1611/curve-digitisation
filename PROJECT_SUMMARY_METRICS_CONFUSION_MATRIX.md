# Curve Digitisation: Full Project Summary, Metrics, and Confusion Matrix

## 1) Project Purpose
This project digitizes performance-curve charts from images and converts visual curves into machine-readable numeric data.

It combines:
- Vision-based metadata extraction (axis bounds, labels, curve descriptors)
- Classical computer vision for pixel extraction and curve tracing
- Coordinate mapping from pixel-space to axis-space
- Curve fitting and quality evaluation
- Multiple interfaces (Streamlit UI, FastAPI, CLI)

Primary outputs are:
- Digitized coordinates per curve
- Fitted equations and quality statistics
- Debug/reconstruction artifacts (overlay, binary, skeleton)
- JSON/CSV exports

---

## 2) Main Entry Points

- Streamlit UI: [streamlit_app.py](streamlit_app.py)
- FastAPI backend: [api.py](api.py)
- Chat CLI: [main.py](main.py)
- Junction-aware CLI: [digitize.py](digitize.py)

Core orchestration:
- Unified pipeline: [core/pipeline.py](core/pipeline.py)

---

## 3) Core Architecture

### 3.1 Metadata and AI Layer
- OpenAI/Azure client for:
  - Intent classification
  - Axis extraction from image
  - Curve feature extraction from image
- File: [core/openai_client.py](core/openai_client.py)

### 3.2 Image Processing Layer
- Main digitizer and routing:
  - Auto mode classification (color vs bw)
  - Plot area detection
  - Curve extraction and fitting
- File: [core/image_processor.py](core/image_processor.py)

- Mode router:
  - HSV saturation heuristics to route to color or bw pipeline
- File: [core/router.py](core/router.py)

- B/W skeleton pipeline:
  - Preprocessing (thresholding, line/text/tick cleanup)
  - Skeletonization
  - Component selection and tracing
  - Dashed/surge handling
- File: [core/bw_pipeline.py](core/bw_pipeline.py)

- Junction-aware advanced CV path (OpenCV-heavy):
  - Hough lines, inpainting, graph-based tracing, junction disambiguation
- File: [core/junction_digitizer.py](core/junction_digitizer.py)

### 3.3 Mapping, Reconstruction, Metrics
- Axis calibration and transforms: [core/calibration.py](core/calibration.py)
- Affine helpers: [core/scale.py](core/scale.py)
- Reconstruction and masks: [core/reconstruction.py](core/reconstruction.py)
- Metric computation: [core/metrics.py](core/metrics.py)
- Typed result models: [core/types.py](core/types.py)

---

## 4) End-to-End Data Flow

1. User uploads image (UI/API/CLI).
2. Vision model extracts axis metadata + curve descriptors.
3. Local CV digitizer loads image and routes mode (color or bw).
4. Plot area is detected/refined.
5. Curve pixels are extracted (color thresholds or bw skeleton path).
6. Pixel coordinates are mapped to axis coordinates.
7. Curves are fitted (polynomial/spline depending on path).
8. Metrics are computed.
9. Artifacts and JSON/CSV outputs are saved to output folder.

---

## 5) Interfaces and Responsibilities

### Streamlit
- Interactive UI, anchor selection, controls, and visualization
- Good for iterative tuning
- File: [streamlit_app.py](streamlit_app.py)

### FastAPI
- Programmatic endpoint for upload and analysis
- Calls unified pipeline and returns structured response
- File: [api.py](api.py)

### Chat CLI
- Conversational interaction and optional image processing
- File: [main.py](main.py)

### Junction CLI
- Specialized B/W junction-aware path with calibration and anchors
- File: [digitize.py](digitize.py)

---

## 6) Output Artifacts
Typical run folder under output:
- input image copy
- digitized curves image
- overlay image
- binary mask image
- skeleton image
- curve_digitization.json

Example run folder:
- [output/20260313_101505](output/20260313_101505)
- Example JSON: [output/20260313_101505/curve_digitization.json](output/20260313_101505/curve_digitization.json)

---

## 7) Metrics: Definitions

Primary per-curve/overall fields in this project:
- delta_value: mean absolute error between extracted points and fitted curve
- delta_norm: normalized delta_value by Y-axis range
- delta_p95: 95th percentile absolute error
- iou: overlap of fitted-vs-actual occupancy bands
- precision: fitted occupied bins that overlap actual
- recall: actual occupied bins captured by fitted

Type definition source:
- [core/types.py](core/types.py#L142)

Metric calculations:
- Global/self-consistency metrics: [core/metrics.py](core/metrics.py#L37)
- Per-curve fit metrics: [core/image_processor.py](core/image_processor.py#L1423)

---

## 8) Real Example Metrics (from Your Run)

Source run:
- [output/20260313_101505/curve_digitization.json](output/20260313_101505/curve_digitization.json)

### 8.1 Overall Metrics
(From overall_metrics block)

| Scope | delta_value | delta_norm | iou | precision | recall | delta_p95 | curve_count |
|---|---:|---:|---:|---:|---:|---:|---:|
| Overall | 14.2351 | 0.001424 | 0.6213 | 0.6233 | 0.9815 | 266.4879 | 6 |

Important clarification:
- This "Overall" row is not computed from one global image-level confusion matrix.
- It is an aggregate across curves in [core/image_processor.py](core/image_processor.py#L2153):
  - delta_value, delta_norm, iou, precision, recall are means over valid curves
  - delta_p95 is the maximum across curves
  - curve_count is number of curves included in that aggregation

Reference lines:
- [output/20260313_101505/curve_digitization.json](output/20260313_101505/curve_digitization.json#L18579)

### 8.2 Per-Curve Example
Curve: gray_0 (label: 11900 RPM)

| Curve | delta_value | delta_norm | iou | precision | recall | delta_p95 |
|---|---:|---:|---:|---:|---:|---:|
| gray_0 (11900 RPM) | 41.3152 | 0.004132 | 0.3077 | 0.3200 | 0.8889 | 266.4879 |

Reference lines:
- [output/20260313_101505/curve_digitization.json](output/20260313_101505/curve_digitization.json#L2965)

---

## 9) Confusion Matrix Interpretation for This Project

Important: this project computes overlap in x-bins for curve-vs-fit matching, not classic class labels.

From implementation:
- intersection_count behaves like TP
- fitted_bins_occupied behaves like TP + FP
- actual_bins_occupied behaves like TP + FN

Reference:
- [core/image_processor.py](core/image_processor.py#L1501)
- [core/image_processor.py](core/image_processor.py#L1532)

Given gray_0 metrics:
- precision = 0.32
- recall = 0.8889
- iou = 0.3077

Why counts can look "small" even when you have 1000+ points:
- The metric compresses points into a fixed number of x-bins (`n_bins = 50`) in [core/image_processor.py](core/image_processor.py#L1492).
- So TP/FP/FN are bin counts (max around 50), not raw point counts.
- 1000+ points can still map into only 50 bins.

Critical correction:
- The earlier TP=16, FP=34, FN=2 table is only one illustrative integer solution from rounded metrics.
- It is not a unique or exact ground-truth count from JSON.
- TN is not explicitly defined by this formulation, because the code tracks only occupied bins (actual and fitted occupancy), not the full negative universe.

### 9.1 Confusion Matrix Table

Per-curve bin-level relation used by code (for each curve):

| Quantity | Definition in code |
|---|---|
| TP-like | intersection_count |
| FP-like | fitted_bins_occupied - intersection_count |
| FN-like | actual_bins_occupied - intersection_count |
| TN | Not computed by current implementation |

### 9.2 Consistency Check

The implemented formulas are:
- precision = intersection_count / fitted_bins_occupied
- recall = intersection_count / actual_bins_occupied
- iou = intersection_count / (actual_bins_occupied + fitted_bins_occupied - intersection_count)

For gray_0, you should interpret the reported numbers as overlap ratios in bin space, not point-space confusion counts.

If exact integer TP/FP/FN/TN is required, the code should be extended to export:
- n_bins
- intersection_count
- fitted_bins_occupied
- actual_bins_occupied
- a defined negative universe (to compute TN)

---

## 10) Practical Reading Guide for Your Metrics

- High recall + lower precision means the fitted curve captures most true curve bins but includes extra bins (over-coverage).
- High delta_p95 with moderate delta_value indicates most fit is okay but worst regions are poor (often near crossings/endpoints/noisy zones).
- Compare overall and per-curve together: one weak curve can dominate p95 while average overlap remains acceptable.

For run 20260313_101505:
- recall is very high (0.9815 overall)
- precision/iou are moderate (0.6233 / 0.6213)
- delta_p95 is high (266.4879), indicating difficult local segments

---

## 11) Suggested Next Evaluation Additions

If needed, the project can be extended with:
- explicit TN definition by adding a full mask-based occupancy universe
- F1 score and balanced accuracy in outputs
- per-curve confusion matrix export directly into JSON
- run-to-run benchmark summary table across output folders

---

## 12) Key File Index

- [README.md](README.md)
- [api.py](api.py)
- [streamlit_app.py](streamlit_app.py)
- [main.py](main.py)
- [digitize.py](digitize.py)
- [core/pipeline.py](core/pipeline.py)
- [core/openai_client.py](core/openai_client.py)
- [core/image_processor.py](core/image_processor.py)
- [core/router.py](core/router.py)
- [core/bw_pipeline.py](core/bw_pipeline.py)
- [core/junction_digitizer.py](core/junction_digitizer.py)
- [core/metrics.py](core/metrics.py)
- [core/types.py](core/types.py)
- [output/20260313_101505/curve_digitization.json](output/20260313_101505/curve_digitization.json)

---

Prepared on 2026-03-13 for the current workspace.