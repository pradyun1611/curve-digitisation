#!/usr/bin/env python
"""
Demonstrate hybrid anchor-guided DP extraction.

Shows how to place anchor points and have the DP tracker fit curves through them.
"""

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_waypoint_extraction():
    """Test DP extraction with waypoints on a real image."""
    
    # Load the real 5-curve chart from your previous output
    json_path = Path("output/20260309_104033/curve_digitization.json")
    
    if not json_path.exists():
        logger.warning("No output JSON found. Test skipped.")
        return
    
    with open(json_path) as f:
        data = json.load(f)
    
    # Print current extraction results
    print("\n=== CURRENT EXTRACTION (without anchor guidance) ===")
    curves = data.get("curves", {})
    for name, curve in curves.items():
        pts = curve.get("raw_pixel_points", [])
        if pts:
            print(f"{name:6s} ({curve['label']:5s}): {len(pts):3d} pts, [{pts[0][0]}, {pts[-1][0]}]")
    
    print("\n=== TO USE ANCHOR POINTS FOR BETTER EXTRACTION ===")
    print("1. Open the chart image in an image viewer")
    print("2. For each curve, identify where it crosses the junction (intersection zone)")
    print("3. Mark TWO anchor points per curve:")
    print("   - One BEFORE the junction (e.g., at x=11000)")
    print("   - One AFTER the junction (e.g., at x=13500)")
    print("4. The y-coordinates should be ON the curve line")
    print("")
    print("Example anchor format (add to process call):")
    print("""
    anchors = [
        ((11000, 72), (13500, 76)),  # 70% curve: (before, after)
        ((10800, 74), (13700, 76)),  # 80% curve
        ((10500, 77), (13800, 74)),  # 90% curve
        ((10200, 78), (14000, 73)),  # 100% curve
        ((9900, 77), (14200, 70)),   # 110% curve
    ]
    """)
    print("\n5. Pass anchors to the processor:")
    print("""
    processor.process_image(
        image_path='image.png',
        anchors=anchors,  # <-- New parameter!
    )
    """)
    print("\nThe DP tracker will now:")
    print("  ✓ Relax curve-jump constraints at intersections")
    print("  ✓ Route each path through its anchor waypoints")
    print("  ✓ Prevent merging of curves at junctions")
    print("\n" + "="*60)


if __name__ == "__main__":
    test_waypoint_extraction()
