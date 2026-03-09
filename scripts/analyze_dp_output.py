#!/usr/bin/env python
"""Analyze the DP tracker output (before resolver) to understand the problem."""
import sys, pathlib
sys.path.insert(0, '.')
import logging
logging.basicConfig(level=logging.WARNING)

import numpy as np
from PIL import Image
from core.bw_pipeline import preprocess_bw, DEFAULT_CONFIG
from core.dp_tracker import extract_curves_dp, _estimate_stroke_width

fpath = sorted(pathlib.Path('input/clear/grayscale').glob('*.png'))[-1]
print(f"Image: {fpath.name}\n")

img = Image.open(fpath)
w, h = img.size
skeleton, binary, _ = preprocess_bw(img, (0,0,w,h), config=DEFAULT_CONFIG)
sw = _estimate_stroke_width(binary)
dp_curves, _ = extract_curves_dp(binary, skeleton, num_curves=3)

img_arr = np.array(img.convert('RGB'))
gray = np.mean(img_arr[:h, :w, :3].astype(np.float32), axis=2).astype(np.uint8)

# For each curve, analyze the shades
print("DP Tracker Output (before resolver):")
for cid, pts in sorted(dp_curves.items()):
    xs = sorted(set(p[0] for p in pts))
    xs_left = [x for x in xs if x < 189]  # before intersection zone
    xs_right = [x for x in xs if x > 328]  # after intersection zone
    xs_zone = [x for x in xs if 189 <= x <= 328]
    
    vals_left = []
    vals_right = []
    vals_zone = []
    for x, y in pts:
        if 0 <= y < h and 0 <= x < w:
            shade = int(gray[y, x])
            if x in xs_left:
                vals_left.append(shade)
            elif x in xs_right:
                vals_right.append(shade)
            elif x in xs_zone:
                vals_zone.append(shade)
    
    left_med = int(np.median(vals_left)) if vals_left else 0
    right_med = int(np.median(vals_right)) if vals_right else 0
    zone_med = int(np.median(vals_zone)) if vals_zone else 0
    
    print(f"  Curve {cid}: {len(pts)} pts, x=[{min(xs)}-{max(xs)}]")
    print(f"    LEFT (before zone): {len(vals_left)} samples, median={left_med}")
    print(f"    ZONE (intersection): {len(vals_zone)} samples, median={zone_med}")
    print(f"    RIGHT (after zone): {len(vals_right)} samples, median={right_med}")
    if left_med >  0 and right_med > 0 and left_med != right_med:
        print(f"    ^^^^ MISMATCH: curve is SPLIT between two real curves!")
