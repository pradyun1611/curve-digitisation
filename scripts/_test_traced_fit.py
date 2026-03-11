"""Test _build_traced_fit method."""
from core.image_processor import CurveDigitizer
import numpy as np

axis_info = {"xMin": 10000, "xMax": 34000, "yMin": 0, "yMax": 3500}
d = CurveDigitizer(axis_info)

# Simulated traced pixels along a diagonal line
pixels = [(100 + i*2, 400 - i) for i in range(200)]
plot_area = (80, 30, 580, 430)

# Normalize to axis coords
axis_coords = d.normalize_to_axis(pixels, 600, 460, plot_area)

# Build traced fit
fit, cleaned = d._build_traced_fit(axis_coords, pixels)
pts = fit["fitted_points"]
xs = [p["x"] for p in pts]
ys = [p["y"] for p in pts]

print(f"Method: {fit['fit_method']}")
print(f"Points: {len(pts)}")
print(f"X range: {min(xs):.0f} - {max(xs):.0f}")
print(f"Y range: {min(ys):.0f} - {max(ys):.0f}")
print(f"R2: {fit['r_squared']:.4f}")
print(f"Peak: {fit['peak_point']}")

# Verify the curve spans the full traced range
assert len(pts) >= 100, f"Too few points: {len(pts)}"
assert max(xs) > 20000, f"X max too low: {max(xs)}"
assert min(ys) > 0, f"Y min negative: {min(ys)}"
assert max(ys) > 1000, f"Y max too low: {max(ys)}"
assert fit["fit_method"] == "traced", f"Wrong method: {fit['fit_method']}"
print("\nAll assertions PASSED")
