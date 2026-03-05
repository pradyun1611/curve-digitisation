# Curve Digitization — Presentation Cheat Sheet

## 1) 30-Second Intro (Say this first)
"This project takes a chart image and converts curve lines into real numeric coordinates. We use AI to read axis limits and curve labels, then computer vision to trace the curves, map pixels to real axis values, fit smooth curves, score quality, and export everything in JSON/CSV for analysis."

---

## 2) One-Line Memory Formula
**P-A-T-H-F-M-S**
- **P**ick image
- **A**sk AI (axis + curve features)
- **T**race curve pixels
- **H**uman units mapping (pixel → axis)
- **F**it smooth curve
- **M**easure quality
- **S**ave outputs

Use this if you forget your flow during Q&A.

---

## 3) End-to-End Workflow (Kid-Friendly)
Think of this like a drawing robot:
1. You show robot a chart photo.
2. Robot asks a teacher:
   - "What are axis min/max and units?"
   - "How many curves and what labels/colors?"
3. Robot decides road:
   - **Color road** if image is colorful.
   - **B/W road** if image is black/white scan.
4. Robot collects curve dots (pixel points).
5. Robot converts dots from pixel world to real chart world.
6. Robot cleans noise and draws a smooth math curve.
7. Robot checks quality (how close fit is to extracted points).
8. Robot saves result files + JSON + CSV.

---

## 4) Exactly What Happens After You Upload an Image
### Streamlit user path
1. Upload image and type prompt, click **Send**.
2. Sidebar settings are read (mode, anchors, thresholds, overrides).
3. AI extracts:
   - axis info (`xMin/xMax/yMin/yMax`, units)
   - curve feature list
4. Core engine processes image:
   - detects plot area rectangle
   - routes to color or B/W extraction
   - extracts curve points
   - maps to axis coordinates
   - fits polynomial
   - computes per-curve and overall metrics
5. App displays:
   - input vs output
   - per-curve metrics and equation
   - downloadable JSON/CSV

---

## 5) Architecture — Which File Does What
### Entrypoints
- CLI chat app: [main.py](main.py)
- Streamlit UI: [streamlit_app.py](streamlit_app.py)
- REST API: [api.py](api.py)

### Core engine
- AI calls (intent, axis, curve features): [core/openai_client.py](core/openai_client.py)
- Color/BW mode classifier: [core/router.py](core/router.py)
- Main digitization pipeline (extract, normalize, fit, graphs): [core/image_processor.py](core/image_processor.py)
- Advanced B/W skeleton tracing pipeline: [core/bw_pipeline.py](core/bw_pipeline.py)
- Pixel↔data calibration and mapping logic: [core/calibration.py](core/calibration.py)
- Quality metrics (IoU, precision, recall, delta, SSIM): [core/metrics.py](core/metrics.py)
- Overlay + reconstructed plot generation: [core/reconstruction.py](core/reconstruction.py)
- Unified API pipeline object/report path: [core/pipeline.py](core/pipeline.py)
- Result dataclasses/schema objects: [core/types.py](core/types.py)

### UI helpers
- Sidebar controls: [ui/sidebar.py](ui/sidebar.py)
- Click-to-place anchor points: [ui/click_canvas.py](ui/click_canvas.py)
- Result rendering + exports: [ui/result_display.py](ui/result_display.py)

---

## 6) Color Path vs B/W Path (Explain clearly)
### Color path
- Extracts pixels by color masks (HSV/RGB rules).
- Removes disconnected noise blobs.
- Converts pixel points to axis points.
- Fits degree-2 polynomial.

### B/W path
- Converts to binary, removes text/ticks/grid noise.
- Skeletonizes lines (1-pixel wide paths).
- Scores components for dashed/text-likeness.
- Uses anchors (if given) for guided tracing.
- Extends endpoints + smooths curve.
- Converts to axis points and fits polynomial.

---

## 7) The Most Important Concept: Plot Area + Mapping
**Plot area** = actual chart rectangle inside whole image.

Why needed:
- Full image includes margins/text/legend.
- Curve points belong to plot area only.

So mapping uses:
- plot-area pixel bounds (`left, top, right, bottom`)
- axis bounds (`xMin, xMax, yMin, yMax`)

Simple explanation line:
"We first localize the graph box, then scale each curve pixel from that box into real engineering units."

---

## 8) JSON Cheat Sheet (Use this in demo)
Reference sample: [output/20260304_091758/curve_digitization.json](output/20260304_091758/curve_digitization.json)

### Top-level fields
- `image_path`: original uploaded image path
- `image_dimensions`: width/height in pixels
- `plot_area`: chart rectangle in pixel coordinates
- `axis_info`: axis min/max, units, image description
- `curves`: per-curve detailed data
- `detected_mode`: `color` or `bw`
- `grayscale_mode`: boolean flag
- `extraction_method`: e.g., `skeleton` or `column_scan`
- `overall_metrics`: graph-level average quality
- `output_graphs`: generated image artifact paths
- `instance_dir`: run output folder
- `pipeline_settings`: settings used in this run
- `validation`: mapping/coverage sanity checks

### Per-curve fields (inside `curves`)
- `label`: curve name (example: `110% speed`)
- `raw_pixel_points`: raw extracted points in image pixel space
- `original_point_count`: count before cleanup
- `normalized_point_count`: count after pixel→axis conversion
- `cleaned_point_count`: count after noise filtering
- `fit_result`:
  - `degree`
  - `coefficients`
  - `equation`
  - `r_squared`
  - `fitted_points` (smooth final coordinate series)
- `metrics`:
  - `delta_value`
  - `delta_norm`
  - `delta_p95`
  - `iou`
  - `precision`
  - `recall`

### Explain 3 confusing JSON terms quickly
- **plot_area**: graph box coordinates in pixels.
- **raw_pixel_points**: original traced dots from image before conversion.
- **fitted_points**: smooth final curve points in axis units (best for analysis/export).

---

## 9) Metrics in Plain English
- **R²**: how well polynomial follows extracted points (closer to 1 is better).
- **delta_value**: average curve error in real axis units.
- **delta_norm**: same error normalized by axis range.
- **delta_p95**: near worst-case error (95th percentile).
- **IoU**: overlap quality between extracted and fitted bands.
- **Precision**: how much fitted curve is truly supported by data.
- **Recall**: how much real extracted data was captured by fit.

---

## 10) 60-Second Technical Walkthrough Script
"User uploads an image in Streamlit. Sidebar settings define mode, thresholds, anchors, and overrides. Azure OpenAI extracts axis bounds and curve features. Router selects color or B/W path. The extractor finds curve pixel points and detects the plot area. Pixel coordinates are normalized to axis coordinates using plot-area-aware mapping, then cleaned and fitted with a polynomial. We compute per-curve and overall quality metrics, render overlay/reconstructed plots, and export JSON/CSV with full traceability from raw pixel points to final fitted coordinates."

---

## 11) Common Questions + Strong Answers
### Q1: Why do you need AI if CV already extracts curves?
A: AI is used for semantic understanding (axis ranges/units/labels). CV is used for geometric extraction. Combined, we get both meaning + coordinates.

### Q2: Why not use full image for mapping?
A: Because curves are only inside the plot box. Using full image introduces scale/offset errors. Plot-area-aware mapping is more accurate.

### Q3: Why keep both raw and fitted points?
A: Raw points provide traceability/auditability. Fitted points provide smooth usable data for analysis.

### Q4: How do you handle B/W scanned charts?
A: Skeleton pipeline removes text/grid noise, traces line structure, supports anchor-guided extraction, then maps and fits.

### Q5: What proves output quality?
A: Curve-level and graph-level metrics (`delta`, `IoU`, `precision`, `recall`, `R²`) plus visual overlay artifacts.

---

## 12) Demo Order (Safe Sequence)
1. Open Streamlit app.
2. Show sidebar controls quickly.
3. Upload one known chart.
4. Click send and show results cards.
5. Open one curve details panel (equation + R² + metrics).
6. Open JSON and point to `plot_area`, `raw_pixel_points`, `fit_result`.
7. Download CSV/JSON and conclude.

---

## 13) Last-Line Closing
"This system converts chart images into reliable numeric curve data with explainable steps, quality metrics, and export-ready outputs for engineering workflows."
