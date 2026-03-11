import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "output/20260310_155614/curve_digitization.json"
with open(path) as f:
    d = json.load(f)

pa = d.get("plot_area", {})
print(f"Plot area: {pa}")
print(f"Axis: x=[{d['axis_info']['xMin']}..{d['axis_info']['xMax']}] y=[{d['axis_info']['yMin']}..{d['axis_info']['yMax']}]")
print(f"Curves: {list(d['curves'].keys())}")
print()

for k, v in d["curves"].items():
    if not isinstance(v, dict):
        continue
    fit = v.get("fit_result", {})
    pts = fit.get("fitted_points", [])
    label = v.get("label", "?")
    rp = v.get("raw_pixel_points", [])
    reproj = v.get("reprojection", {})
    
    if pts:
        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        print(f"{k}: label={label}  deg={fit.get('degree')}  r2={fit.get('r_squared',0):.4f}  "
              f"x=[{min(xs):.1f}..{max(xs):.1f}]  y=[{min(ys):.1f}..{max(ys):.1f}]  pts={len(pts)}")
    else:
        print(f"{k}: label={label}  NO FITTED POINTS")
    
    if rp:
        rxs = [p[0] for p in rp]
        rys = [p[1] for p in rp]
        print(f"  raw_pixels: n={len(rp)}  x=[{min(rxs)}..{max(rxs)}]  y=[{min(rys)}..{max(rys)}]")
    
    if reproj:
        print(f"  reprojection: {reproj}")
    print()
