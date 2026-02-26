"""Diagnostic: run grayscale extraction step-by-step on a real image
and dump intermediate data to understand why curves fragment."""
import sys, os
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Find the real input image from the latest output
latest = sorted(_ROOT.glob("output/2026*"))[-1]
# Look for an input_ file in that dir
inp_files = list(latest.glob("input_*"))
if inp_files:
    img_path = inp_files[0]
else:
    img_path = _ROOT / "input" / "image.png"

print(f"Image: {img_path}")
image = Image.open(str(img_path))
width, height = image.size
print(f"Dimensions: {width}x{height}")

img_array = np.array(image)

# Replicate plot_area detection
from core.image_processor import CurveDigitizer
axis_info = {
    "xMin": 0, "xMax": 30000,
    "yMin": 0, "yMax": 3500,
    "xUnit": "kg/hr", "yUnit": "kW",
}
digitizer = CurveDigitizer(axis_info)
plot_area = digitizer.detect_plot_area(image)
print(f"plot_area: {plot_area}")

p_left, p_top, p_right, p_bottom = plot_area
inset = 5
p_left  = min(p_left + inset,  p_right - 1)
p_top   = min(p_top + inset,   p_bottom - 1)
p_right = max(p_right - inset,  p_left + 1)
p_bottom = max(p_bottom - inset, p_top + 1)

region = img_array[p_top:p_bottom, p_left:p_right]
if region.ndim == 3:
    gray = np.mean(region[:, :, :3].astype(np.float32), axis=2).astype(np.uint8)
else:
    gray = region.astype(np.uint8)
region_h, region_w = gray.shape
print(f"Region: {region_w}x{region_h}")

# Denoise
gray_pil = Image.fromarray(gray).filter(ImageFilter.MedianFilter(size=3))
gray = np.array(gray_pil)

# Otsu
threshold = digitizer._otsu_threshold(gray)
print(f"Otsu threshold: {threshold}")

# Histogram of gray values
hist, bins = np.histogram(gray.ravel(), bins=50, range=(0, 255))
print("\nGray histogram (top 15 bins):")
for i in np.argsort(hist)[-15:][::-1]:
    lo, hi = int(bins[i]), int(bins[i+1])
    print(f"  [{lo:3d}-{hi:3d}]: {hist[i]:6d} px")

# Binary with threshold + margin
binary_std = gray <= threshold
binary_margin = gray <= min(threshold + 10, 200)
print(f"\nDark pixels (gray <= {threshold}): {binary_std.sum()}")
print(f"Dark pixels (gray <= {min(threshold+10, 200)}): {binary_margin.sum()}")
binary = binary_margin

# Morphological close
from scipy.ndimage import binary_closing, binary_opening, binary_dilation
from scipy.ndimage import label as ndimage_label
close_kernel = np.ones((3, 5), dtype=bool)
binary = binary_closing(binary, structure=close_kernel, iterations=1)

# Grid suppression
row_fill = binary.sum(axis=1) / region_w
grid_rows = row_fill > 0.75
n_grid_rows = grid_rows.sum()
binary[grid_rows, :] = False
col_fill = binary.sum(axis=0) / region_h
grid_cols = col_fill > 0.75
n_grid_cols = grid_cols.sum()
binary[:, grid_cols] = False
print(f"Grid rows removed: {n_grid_rows}, grid cols removed: {n_grid_cols}")
print(f"Dark pixels after grid removal: {binary.sum()}")

# Opening
open_kernel = np.ones((2, 2), dtype=bool)
binary = binary_opening(binary, structure=open_kernel, iterations=1)
print(f"Dark pixels after opening: {binary.sum()}")

# Dilation
dilate_kernel = np.ones((3, 21), dtype=bool)
dilated = binary_dilation(binary, structure=dilate_kernel, iterations=1)

# Connected components on dilated
structure_8 = np.ones((3, 3), dtype=int)
labelled_dil, n_comp_dil = ndimage_label(dilated, structure=structure_8)
print(f"\nDilated components: {n_comp_dil}")

# Map back to original
labelled = labelled_dil * binary.astype(labelled_dil.dtype)

# Analyze each component
min_width = max(10, int(region_w * 0.05))
print(f"Min width filter: {min_width}px (region_w={region_w})")
print("\nAll components:")
valid = []
for comp_id in range(1, n_comp_dil + 1):
    ys, xs = np.where(labelled == comp_id)
    n_px = len(ys)
    if n_px < 5:
        continue
    h_extent = int(xs.max()) - int(xs.min()) + 1
    v_extent = int(ys.max()) - int(ys.min()) + 1
    aspect = h_extent / max(v_extent, 1)
    mean_y = float(np.mean(ys))
    
    # Check filter conditions
    skip_reason = ""
    if h_extent < min_width:
        skip_reason = f"h_extent={h_extent} < min_width={min_width}"
    elif h_extent > region_w * 0.85 and aspect > 40:
        skip_reason = f"full-width thin h-line"
    elif v_extent > region_h * 0.85 and aspect < 0.03:
        skip_reason = f"full-height thin v-line"
    
    status = "SKIP" if skip_reason else "KEEP"
    print(f"  comp {comp_id}: {n_px:5d}px, h_ext={h_extent:4d}, v_ext={v_extent:4d}, "
          f"aspect={aspect:6.1f}, mean_y={mean_y:6.1f} -> {status}")
    if skip_reason:
        print(f"    reason: {skip_reason}")
    else:
        valid.append((comp_id, mean_y, n_px, h_extent, v_extent))

valid.sort(key=lambda x: x[1])
print(f"\nValid components: {len(valid)}")
for i, (cid, my, npx, hw, vw) in enumerate(valid):
    print(f"  [{i}] comp={cid}: {npx}px, h_extent={hw}, v_extent={vw}, mean_y={my:.1f}")

# Save debug images
dbg_dir = _ROOT / "tests" / "artifacts" / "real_bw"
dbg_dir.mkdir(parents=True, exist_ok=True)

Image.fromarray(gray).save(str(dbg_dir / "01_gray_region.png"))
Image.fromarray((binary * 255).astype(np.uint8)).save(str(dbg_dir / "02_binary.png"))
Image.fromarray((dilated * 255).astype(np.uint8)).save(str(dbg_dir / "03_dilated.png"))

# Color-coded components
comp_img = np.zeros((*gray.shape, 3), dtype=np.uint8)
colors = [(255,0,0),(0,180,0),(0,0,255),(255,165,0),(128,0,255),(0,200,200),(255,255,0)]
for i, (cid, _, _, _, _) in enumerate(valid):
    ys, xs = np.where(labelled == cid)
    c = colors[i % len(colors)]
    comp_img[ys, xs] = c
Image.fromarray(comp_img).save(str(dbg_dir / "04_components.png"))

print(f"\nDebug images saved to: {dbg_dir}")
