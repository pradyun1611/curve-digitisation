"""Diagnose: analyze each track's properties to identify surge/dashed lines."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageFilter
from core.image_processor import CurveDigitizer

# Find real image from latest output with input_
root = Path(__file__).resolve().parent.parent
for d in sorted(root.glob("output/2026*"), reverse=True):
    inp = [p for p in d.iterdir() if p.name.startswith("input_")]
    if inp:
        img_path = inp[0]
        break

print(f"Image: {img_path.name}")
image = Image.open(str(img_path))
width, height = image.size

axis_info = {"xMin": 0, "xMax": 30000, "yMin": 0, "yMax": 3500,
             "xUnit": "kg/hr", "yUnit": "kW"}
digitizer = CurveDigitizer(axis_info)
plot_area = digitizer.detect_plot_area(image)
print(f"plot_area: {plot_area}")

# Run extraction with high num_curves
gray_clusters = digitizer.extract_curves_grayscale(image, 10, plot_area)
print(f"\nTracks found: {len(gray_clusters)}")

p_left, p_top, p_right, p_bottom = plot_area
region_w = p_right - p_left
region_h = p_bottom - p_top

for idx, points in sorted(gray_clusters.items()):
    pts = np.array(points)
    xs = pts[:, 0]
    ys = pts[:, 1]
    n = len(pts)
    
    # X span (in region coords)
    x_span = int(xs.max()) - int(xs.min())
    x_span_ratio = x_span / region_w
    
    # Y span
    y_span = int(ys.max()) - int(ys.min())
    
    # Slope: overall dy/dx
    if x_span > 0:
        slope = y_span / x_span
    else:
        slope = float('inf')
    
    # Coverage density: what fraction of columns in the x-range have a point
    unique_x = len(set(xs.tolist()))
    coverage = unique_x / max(x_span, 1)
    
    # Gap analysis: max gap between consecutive x values
    x_sorted = sorted(set(xs.tolist()))
    if len(x_sorted) > 1:
        gaps = [x_sorted[i+1] - x_sorted[i] for i in range(len(x_sorted)-1)]
        max_gap = max(gaps)
        mean_gap = np.mean(gaps)
        n_large_gaps = sum(1 for g in gaps if g > 5)
    else:
        max_gap = 0
        mean_gap = 0
        n_large_gaps = 0
    
    # Classify
    is_steep = slope > 2.0  # surge lines are nearly vertical
    is_dashed = coverage < 0.5 or n_large_gaps > len(x_sorted) * 0.3
    
    flags = []
    if is_steep:
        flags.append("STEEP/SURGE")
    if is_dashed:
        flags.append("DASHED")
    
    print(f"\n  Track {idx}: {n} pts")
    print(f"    x=[{int(xs.min())}, {int(xs.max())}] span={x_span}px ({x_span_ratio:.1%} of plot)")
    print(f"    y=[{int(ys.min())}, {int(ys.max())}] span={y_span}px")
    print(f"    slope={slope:.2f}, coverage={coverage:.2f}")
    print(f"    unique_x={unique_x}, max_gap={max_gap}, mean_gap={mean_gap:.1f}, large_gaps={n_large_gaps}")
    if flags:
        print(f"    ** FLAGS: {', '.join(flags)} **")
