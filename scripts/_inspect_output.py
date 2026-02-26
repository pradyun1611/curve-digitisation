"""Quick inspection of the latest output JSON."""
import json, sys, glob
from pathlib import Path

root = Path(__file__).resolve().parent.parent
# Find latest output folder
dirs = sorted(root.glob("output/2026*"))
if not dirs:
    print("No output dirs found"); sys.exit(1)

latest = dirs[-1]
jpath = latest / "curve_digitization.json"
if not jpath.exists():
    print(f"No JSON in {latest}"); sys.exit(1)

with open(jpath) as f:
    d = json.load(f)

print(f"Dir: {latest.name}")
print(f"grayscale_mode: {d.get('grayscale_mode')}")
print(f"image_dims: {d.get('image_dimensions')}")
print(f"plot_area: {d.get('plot_area')}")
print(f"axis_info: {d.get('axis_info')}")
curves = d.get("curves", {})
print(f"num_curves: {len(curves)}")
for k, v in curves.items():
    if not isinstance(v, dict):
        continue
    fit = v.get("fit_result", {})
    pts = fit.get("fitted_points", [])
    label = v.get("label", "?")
    mode = v.get("extraction_mode", "?")
    orig = v.get("original_point_count", 0)
    cleaned = v.get("cleaned_point_count", 0)
    deg = fit.get("degree")
    r2 = fit.get("r_squared")
    err = v.get("error", "")
    print(f"  {k}: label={label}, mode={mode}, orig_pts={orig}, "
          f"cleaned={cleaned}, fitted={len(pts)}, degree={deg}, R2={r2}")
    if err:
        print(f"    ERROR: {err}")
    if pts:
        ys = [p["y"] for p in pts]
        xs = [p["x"] for p in pts]
        print(f"    x=[{min(xs):.1f}, {max(xs):.1f}] y=[{min(ys):.1f}, {max(ys):.1f}]")

print(f"\noverall_metrics: {d.get('overall_metrics')}")
