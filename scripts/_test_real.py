"""Test column-scan extraction on the real image."""
import sys, numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from PIL import Image
from core.image_processor import CurveDigitizer

# Find real image from latest output
latest = sorted(Path("output").glob("2026*"))[-1]
img_path = next(p for p in latest.iterdir() if p.name.startswith("input_"))

axis_info = {"xMin": 0, "xMax": 30000, "yMin": 0, "yMax": 3500,
             "xUnit": "kg/hr", "yUnit": "kW"}
digitizer = CurveDigitizer(axis_info)
features = {"curves": [
    {"color": "black", "shape": "curved", "label": f"{pct}% SPEED"}
    for pct in [110, 100, 90, 80, 70]
]}

import tempfile
with tempfile.TemporaryDirectory() as tmp:
    results = digitizer.process_curve_image(str(img_path), features, tmp, mode="grayscale")

curves = results.get("curves", {})
print(f"Curves found: {len(curves)}")
for k, v in curves.items():
    if not isinstance(v, dict):
        continue
    fit = v.get("fit_result", {})
    pts = fit.get("fitted_points", [])
    err = v.get("error", "")
    if err:
        print(f"  {k}: ERROR={err}")
        continue
    ys = [p["y"] for p in pts] if pts else []
    xs = [p["x"] for p in pts] if pts else []
    label = v.get("label", "?")
    orig = v.get("original_point_count", 0)
    cleaned = v.get("cleaned_point_count", 0)
    deg = fit.get("degree")
    r2 = fit.get("r_squared", 0)
    print(f"  {k}: label={label}, orig={orig}, cleaned={cleaned}, "
          f"fitted={len(pts)}, deg={deg}, R2={r2:.4f}")
    if xs:
        print(f"    x=[{min(xs):.0f}, {max(xs):.0f}] y=[{min(ys):.0f}, {max(ys):.0f}]")
