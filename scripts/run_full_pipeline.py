#!/usr/bin/env python
"""Run the full B/W pipeline on a single grayscale image and show results."""
import sys, pathlib
sys.path.insert(0, '.')

import logging
logging.basicConfig(level=logging.WARNING)

from PIL import Image
from core.bw_pipeline import extract_bw_curves, DEFAULT_CONFIG

# Test Image 3
fpath = sorted(pathlib.Path('input/clear/grayscale').glob('*.png'))[-1]
print(f"Image: {fpath.name}")
img = Image.open(fpath)
w, h = img.size
plot_area = (0, 0, w, h)

# Run the full pipeline (includes resolver)
curves = extract_bw_curves(img, num_curves=3, plot_area=plot_area, config=DEFAULT_CONFIG)
print(f"\nExtracted {len(curves)} curves:")
for cid, pts in sorted(curves.items()):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    n_q = len(pts) // 4
    median_shades = []
    gray = [p for p in img.convert('L').getdata()]
    w_gray = img.width
    for qi in range(4):
        start = qi * n_q
        end = (qi + 1) * n_q
        chunk_pts = sorted(pts[start:end], key=lambda p: p[0])
        vals = []
        for x, y in chunk_pts:
            if 0 <= y < h and 0 <= x < w:
                vals.append(gray[y * w + x])
        median_shades.append(int(__import__('statistics').median(vals)) if vals else 0)
    print(f"  Curve {cid}: {len(pts)} pts, x=[{min(xs)}-{max(xs)}], "
          f"y=[{min(ys)}-{max(ys)}], shades_Q={median_shades}")
