"""Diagnose: what's causing large delta values on real images."""
import sys, numpy as np, tempfile, json
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

# Test on latest real image
for d in sorted(root.glob("output/2026*"), reverse=True)[:3]:
    inp = [p for p in d.iterdir() if p.name.startswith("input_")]
    if not inp:
        continue
    img_path = inp[0]
    print(f"\n{'='*60}")
    print(f"Image: {img_path.name}")
    
    dig = CurveDigitizer(axis_info)
    with tempfile.TemporaryDirectory() as tmp:
        results = dig.process_curve_image(str(img_path), features, tmp, mode="grayscale")
    
    print(f"Overall metrics: {results.get('overall_metrics', {})}")
    
    curves = results.get("curves", {})
    for k, v in curves.items():
        if not isinstance(v, dict) or v.get("error"):
            continue
        fit = v.get("fit_result", {})
        pts = fit.get("fitted_points", [])
        metrics = v.get("metrics", {})
        label = v.get("label", "?")
        orig = v.get("original_point_count", 0)
        cleaned = v.get("cleaned_point_count", 0)
        r2 = fit.get("r_squared", 0)
        
        # Compute residuals between cleaned points and fitted curve
        # The metrics dict should have delta_value, delta_norm, etc.
        print(f"\n  {k} ({label}):")
        print(f"    orig={orig}, cleaned={cleaned}, fitted={len(pts)}, R2={r2:.4f}")
        print(f"    metrics: delta_value={metrics.get('delta_value')}, "
              f"delta_norm={metrics.get('delta_norm')}, "
              f"delta_p95={metrics.get('delta_p95')}")
        print(f"    iou={metrics.get('iou')}, prec={metrics.get('precision')}, "
              f"recall={metrics.get('recall')}")
        
        if pts:
            ys = [p["y"] for p in pts]
            xs = [p["x"] for p in pts]
            print(f"    fitted x=[{min(xs):.0f},{max(xs):.0f}] y=[{min(ys):.0f},{max(ys):.0f}]")
        
        # Check raw pixel points mapping
        raw_px = v.get("raw_pixel_points", [])
        if raw_px:
            raw_arr = np.array(raw_px)
            print(f"    raw_pixel x=[{raw_arr[:,0].min()},{raw_arr[:,0].max()}] "
                  f"y=[{raw_arr[:,1].min()},{raw_arr[:,1].max()}]")
