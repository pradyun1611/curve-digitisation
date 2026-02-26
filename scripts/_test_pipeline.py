"""Full pipeline test on real images after filtering fix."""
import sys, numpy as np, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.image_processor import CurveDigitizer

root = Path(__file__).resolve().parent.parent
axis_info = {"xMin": 0, "xMax": 30000, "yMin": 0, "yMax": 3500,
             "xUnit": "kg/hr", "yUnit": "kW"}

features = {"curves": [
    {"color": "black", "shape": "curved", "label": f"{pct}% SPEED"}
    for pct in [110, 100, 90, 80, 70]
]}

for d in sorted(root.glob("output/2026*"), reverse=True)[:5]:
    inp = [p for p in d.iterdir() if p.name.startswith("input_")]
    if not inp:
        continue
    img_path = inp[0]
    print(f"\n{'='*60}")
    print(f"Image: {img_path.name}")
    
    dig = CurveDigitizer(axis_info)
    with tempfile.TemporaryDirectory() as tmp:
        results = dig.process_curve_image(str(img_path), features, tmp, mode="grayscale")
    
    curves = results.get("curves", {})
    print(f"Curves: {len(curves)}")
    for k, v in curves.items():
        if not isinstance(v, dict):
            continue
        fit = v.get("fit_result", {})
        pts = fit.get("fitted_points", [])
        err = v.get("error", "")
        if err:
            print(f"  {k}: ERROR={err}")
            continue
        ys = [p["y"] for p in pts]
        xs = [p["x"] for p in pts]
        r2 = fit.get("r_squared", 0)
        label = v.get("label", "?")
        print(f"  {k}: label={label}, pts={len(pts)}, R2={r2:.4f}, "
              f"x=[{min(xs):.0f},{max(xs):.0f}] y=[{min(ys):.0f},{max(ys):.0f}]")
