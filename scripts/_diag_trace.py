"""Quick diagnostic: trace the grayscale pipeline output."""
import sys, tempfile, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
from core.image_processor import CurveDigitizer

axis_info = {"xMin": 0, "xMax": 100, "yMin": 0, "yMax": 100, "xUnit": "%", "yUnit": "%"}
dig = CurveDigitizer(axis_info)
img = Image.open(str(ROOT / "tests" / "data" / "input_bw.png")).convert("RGB")
w, h = img.size
plot_area = dig.detect_plot_area(img)
print(f"Image: {w}x{h}, plot_area: {plot_area}")

# --- Raw extraction ---
clusters = dig.extract_curves_grayscale(img, 3, plot_area)
print(f"\nextract_curves_grayscale returned {len(clusters)} clusters")
for k, pts in clusters.items():
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    print(f"  cluster {k}: {len(pts)} pts, pixel x=[{min(xs)}-{max(xs)}], pixel y=[{min(ys)}-{max(ys)}]")

# --- Full pipeline ---
features = {"curves": [
    {"color": "black", "shape": "curved", "label": "Curve 1"},
    {"color": "black", "shape": "curved", "label": "Curve 2"},
    {"color": "black", "shape": "curved", "label": "Curve 3"},
]}
with tempfile.TemporaryDirectory() as tmp:
    results = dig.process_curve_image(str(ROOT / "tests" / "data" / "input_bw.png"), features, tmp)

print(f"\nprocess_curve_image returned {len(results.get('curves', {}))} curves")
for k, v in results.get("curves", {}).items():
    fit = v.get("fit_result", {})
    fitted = fit.get("fitted_points", [])
    if fitted:
        ys_fit = [p["y"] for p in fitted]
        xs_fit = [p["x"] for p in fitted]
        print(f"  {k}: label={v.get('label')}, n_fitted={len(fitted)}, "
              f"x=[{min(xs_fit):.1f},{max(xs_fit):.1f}], "
              f"y=[{min(ys_fit):.1f},{max(ys_fit):.1f}], "
              f"R2={fit.get('r_squared', '?')}, deg={fit.get('degree')}")
    else:
        print(f"  {k}: error={v.get('error', 'no fit')}")

# --- Compare: what the COLOR pipeline produces on the same image ---
print("\n--- Color pipeline on SAME image (forced) ---")
with tempfile.TemporaryDirectory() as tmp:
    results_color = dig.process_curve_image(
        str(ROOT / "tests" / "data" / "input_bw.png"), features, tmp, mode="color"
    )
for k, v in results_color.get("curves", {}).items():
    fit = v.get("fit_result", {})
    fitted = fit.get("fitted_points", [])
    if fitted:
        ys_fit = [p["y"] for p in fitted]
        xs_fit = [p["x"] for p in fitted]
        print(f"  {k}: n_fitted={len(fitted)}, "
              f"x=[{min(xs_fit):.1f},{max(xs_fit):.1f}], "
              f"y=[{min(ys_fit):.1f},{max(ys_fit):.1f}], "
              f"R2={fit.get('r_squared', '?')}")
    else:
        print(f"  {k}: n_px={v.get('original_point_count')}, error={v.get('error', 'no fit')}")
