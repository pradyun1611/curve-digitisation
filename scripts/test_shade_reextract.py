#!/usr/bin/env python
"""
Test shade-guided re-extraction on real grayscale images.

Compares resolver OFF vs ON with quarterly shade analysis.
"""
from __future__ import annotations
import sys, pathlib, logging
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from core.bw_pipeline import preprocess_bw, DEFAULT_CONFIG
from core.dp_tracker import extract_curves_dp, _estimate_stroke_width
from core.intersection_resolver import resolve_intersections

logging.basicConfig(level=logging.WARNING, format="%(name)s %(message)s")
# Enable resolver debug logs for Image 2 only 
logging.getLogger("core.intersection_resolver").setLevel(logging.WARNING)
logging.getLogger("core.dp_tracker").setLevel(logging.WARNING)

INPUT_DIR = pathlib.Path("input/clear/grayscale")
files = sorted(INPUT_DIR.glob("*.png"))[-3:]
print(f"\n{'='*70}")
print(f"Testing shade-guided re-extraction on {len(files)} images")
print(f"{'='*70}")


def quarterly_shade(gray: np.ndarray, pts: list) -> list:
    """Return median shade of 4 equal x-quarters."""
    if not pts:
        return [0, 0, 0, 0]
    pts_sorted = sorted(pts, key=lambda p: p[0])
    n = len(pts_sorted)
    qs = []
    for qi in range(4):
        start = qi * n // 4
        end = (qi + 1) * n // 4
        chunk = pts_sorted[start:end]
        vals = []
        for x, y in chunk:
            if 0 <= y < gray.shape[0] and 0 <= x < gray.shape[1]:
                vals.append(int(gray[y, x]))
        qs.append(int(np.median(vals)) if vals else 0)
    return qs


for fpath in files:
    print(f"\n{'-'*70}")
    print(f"Image: {fpath.name}")
    print(f"{'-'*70}")

    img = Image.open(fpath)
    w, h = img.size
    plot_area = (0, 0, w, h)

    skeleton, binary, adj_area = preprocess_bw(img, plot_area, config=DEFAULT_CONFIG)
    rh, rw = skeleton.shape

    img_arr = np.array(img.convert("RGB"))
    gray = np.mean(img_arr[:h, :w, :3].astype(np.float32), axis=2).astype(np.uint8)

    # Extract curves with DP
    sw = _estimate_stroke_width(binary)
    dp_result, _ = extract_curves_dp(binary, skeleton, num_curves=3)

    print(f"  Image size: {w}x{h}, stroke_width={sw}")
    print(f"  DP tracker found {len(dp_result)} curves")

    # --- Resolver OFF ---
    print(f"\n  --- Resolver OFF ---")
    for cid, pts in sorted(dp_result.items()):
        qs = quarterly_shade(gray, pts)
        std_q = np.std(qs)
        print(f"    Curve {cid}: Qs={qs} std={std_q:.1f} ({len(pts)} pts)")

    # --- Resolver ON ---
    print(f"\n  --- Resolver ON ---")
    try:
        # For Image 2, show curve x-ranges vs zone boundaries
        if "114221" in fpath.name:
            from core.intersection_resolver import _find_intersection_zones
            zones = _find_intersection_zones(
                dp_result, gray.shape[0], gray.shape[1],
                proximity_px=0,
                min_zone_width=max(5, sw),
                merge_gap=max(20, sw * 5),
                pad_x=max(30, sw * 8),
            )
            for zi, z in enumerate(zones):
                print(f"    Zone {zi}: x=[{z['x_start']},{z['x_end']}] curves={z['curve_ids']}")
            for cid, pts in sorted(dp_result.items()):
                xs = [p[0] for p in pts]
                print(f"    Curve {cid} x_range=[{min(xs)},{max(xs)}] ({len(xs)} pts)")

        resolved = resolve_intersections(
            gray, binary, dp_result,
            stroke_width=sw,
            proximity_px=DEFAULT_CONFIG.intersection_proximity_px,
            intensity_weight=DEFAULT_CONFIG.intersection_intensity_weight,
            geometry_weight=DEFAULT_CONFIG.intersection_geometry_weight,
            curvature_weight=DEFAULT_CONFIG.intersection_curvature_weight,
            debug_dir="output/debug_shade_test",
        )
        for cid, pts in sorted(resolved.items()):
            qs = quarterly_shade(gray, pts)
            std_q = np.std(qs)
            print(f"    Curve {cid}: Qs={qs} std={std_q:.1f} ({len(pts)} pts)")
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*70}")
print("Done.")
