#!/usr/bin/env python
"""
CLI for junction-aware curve digitizer.

Usage:
  # Auto mode – extract all curves automatically:
  python digitize.py --image path/to/plot.png --out output_dir
  python digitize.py --image plot.png --num-curves 5 --debug

  # Single-curve mode – place anchors on ONE curve, get only that curve:
  python digitize.py --image plot.png --points 50,380 200,300 350,380 --label "70%%"

  # Multi-curve anchored mode via JSON file:
  python digitize.py --image plot.png --anchors anchors.json --out out_dir

  # With calibration:
  python digitize.py --image plot.png --calib calib.json --points 50,380 200,300 350,380

Calibration JSON formats:

  3-point affine (recommended):
    {
      "mode": "3point",
      "points": [
        {"pixel": [100, 400], "data": [0, 0]},
        {"pixel": [500, 400], "data": [200000, 0]},
        {"pixel": [100, 50],  "data": [0, 70]}
      ]
    }

  2-per-axis (after deskew):
    {
      "mode": "2axis",
      "x_refs": [{"pixel": 100, "value": 0}, {"pixel": 500, "value": 200000}],
      "y_refs": [{"pixel": 400, "value": 40}, {"pixel": 50, "value": 70}],
      "plot_area": [100, 50, 500, 400]
    }

Anchors JSON format:
    {
      "curves": [
        {"label": "70%%",  "anchors": [[50, 380], [200, 300], [350, 380]]},
        {"label": "80%%",  "anchors": [[80, 350], [250, 250], [400, 350]]}
      ]
    }
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path so core/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.junction_digitizer import JunctionConfig, digitize, select_anchors_interactive


def main():
    parser = argparse.ArgumentParser(
        description="Junction-aware multi-curve digitizer for scanned B/W plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--calib", default=None, help="Path to calibration JSON")
    parser.add_argument("--anchors", default=None, help="Path to anchors JSON")
    parser.add_argument("--points", nargs="+", default=None,
                        help="Anchor points for a SINGLE curve as x,y pairs "
                             "(e.g. --points 50,380 200,300 350,380)")
    parser.add_argument("--label", default="curve",
                        help="Label for the curve when using --points (default: curve)")
    parser.add_argument("--out", default="output_junction", help="Output directory")
    parser.add_argument("--num-curves", type=int, default=0,
                        help="Expected number of curves (0=auto)")
    parser.add_argument("--debug", action="store_true", help="Save debug images")
    parser.add_argument("--beam-width", type=int, default=5,
                        help="Beam search width for multi-hypothesis tracking")
    parser.add_argument("--lambda-turn", type=float, default=8.0,
                        help="Turn angle penalty weight")
    parser.add_argument("--resample", type=int, default=300,
                        help="Number of resampled points per curve")
    parser.add_argument("--refine", action="store_true", default=True,
                        help="Enable multi-scale zoom junction refinement (default: on)")
    parser.add_argument("--no-refine", dest="refine", action="store_false",
                        help="Disable multi-scale zoom junction refinement")
    parser.add_argument("--refine-roi-radius", type=int, default=60,
                        help="Pixel radius of ROI around each junction for zoom refinement")
    parser.add_argument("--refine-upscale", type=int, default=4,
                        help="Upscale factor for zoomed junction ROI (4-8)")
    parser.add_argument("--anchors-click", action="store_true",
                        help="Interactively click anchors on the image using matplotlib")
    parser.add_argument("--lambda-angle", type=float, default=2.0,
                        help="Turn angle penalty for A* tracing (default: 2.0)")
    parser.add_argument("--lambda-junction", type=float, default=5.0,
                        help="Junction crossing penalty for A* tracing (default: 5.0)")
    parser.add_argument("--snap-radius", type=int, default=30,
                        help="Radius for snapping anchors to skeleton (default: 30)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose logging")

    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Config
    config = JunctionConfig(
        beam_width=args.beam_width,
        lambda_turn=args.lambda_turn,
        lambda_angle=args.lambda_angle,
        lambda_junction=args.lambda_junction,
        snap_radius=args.snap_radius,
        resample_n=args.resample,
        refine_enabled=args.refine,
        refine_roi_radius=args.refine_roi_radius,
        refine_upscale=args.refine_upscale,
        debug=args.debug,
        debug_dir=args.out,
    )

    # Handle --anchors-click / --points: single-curve anchor mode
    anchors_path = args.anchors
    _tmp_anchors_path = None

    if args.anchors_click:
        import cv2 as _cv2
        _img = _cv2.imread(args.image)
        if _img is None:
            parser.error(f"Cannot load image: {args.image}")
        clicked = select_anchors_interactive(_img)
        if len(clicked) < 2:
            parser.error("Need at least 2 anchor points (click more points)")
        import tempfile
        anchors_data = {"curves": [{"label": args.label,
                                    "anchors": [[a[0], a[1]] for a in clicked]}]}
        os.makedirs(args.out, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir=args.out)
        json.dump(anchors_data, tmp)
        tmp.close()
        _tmp_anchors_path = tmp.name
        anchors_path = tmp.name
        print(f"Interactive mode: {len(clicked)} anchors for '{args.label}'")
    elif args.points:
        pts = []
        for p in args.points:
            parts = p.split(",")
            if len(parts) != 2:
                parser.error(f"Invalid point format '{p}' — use x,y (e.g. 200,300)")
            pts.append([int(parts[0]), int(parts[1])])
        if len(pts) < 2:
            parser.error("Need at least 2 anchor points (--points x1,y1 x2,y2 ...)")
        import tempfile
        anchors_data = {"curves": [{"label": args.label, "anchors": pts}]}
        os.makedirs(args.out, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir=args.out)
        json.dump(anchors_data, tmp)
        tmp.close()
        _tmp_anchors_path = tmp.name
        anchors_path = tmp.name
        print(f"Single-curve mode: tracing '{args.label}' with {len(pts)} anchors")

    # Run
    result = digitize(
        image_path=args.image,
        calib_path=args.calib,
        anchors_path=anchors_path,
        output_dir=args.out,
        config=config,
        num_curves=args.num_curves,
    )

    # Clean up temp file
    if _tmp_anchors_path:
        try:
            os.remove(_tmp_anchors_path)
        except OSError:
            pass

    # Print summary
    print("\n" + "=" * 60)
    print("DIGITIZATION RESULTS")
    print("=" * 60)
    print(result["summary"])
    print(f"\nSkew correction: {result['skew_angle']:.2f}°")
    print(f"Graph: {result['num_edges']} edges, {result['num_endpoints']} endpoints, "
          f"{result['num_junctions']} junctions")
    print(f"Junction pairings: {result['junction_pairings_count']}")
    print(f"Junctions refined (multi-scale zoom): {result.get('num_junctions_refined', 0)}")
    print(f"Curves extracted: {len(result['curves'])}")
    if result.get("avg_reprojection_error_px") is not None:
        print(f"Avg reprojection error: {result['avg_reprojection_error_px']:.3f} px")
    if result.get("mapping_error_px") is not None:
        print(f"Calibration round-trip error: {result['mapping_error_px']:.3f} px")
    print(f"\nOutput files:")
    for f in result["files_written"]:
        print(f"  {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
