#!/usr/bin/env python
"""Test helper – run the enhanced BW pipeline on sample images and save debug output.

Usage::

    python -m scripts.test_bw_robust                           # all images in input/
    python -m scripts.test_bw_robust --image input/clear/grayscale/sample.png
    python -m scripts.test_bw_robust --debug                   # enable debug overlays

The script exercises:
  * CLAHE + blackhat + adaptive threshold preprocessing
  * Hough-based axis/gridline removal
  * Graph-based multi-curve extraction
  * Debug overlay generation (--debug flag)

Results are saved under ``output/<job_id>/debug_bw/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from PIL import Image

from core.bw_pipeline import (
    detect_plot_area_robust,
    extract_bw_curves,
    preprocess_bw,
)
from core.config import BWPipelineConfig, DEFAULT_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-5s  %(message)s",
)
logger = logging.getLogger("test_bw_robust")


def _collect_images(root: Path) -> list[Path]:
    """Return all PNG/JPG images under *root*."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)


def _run_one(
    img_path: Path,
    out_dir: Path,
    config: BWPipelineConfig,
    num_curves: int = 2,
) -> dict:
    """Run the enhanced BW pipeline on a single image."""
    logger.info("Processing %s", img_path)
    image = Image.open(img_path).convert("RGB")

    t0 = time.perf_counter()
    plot_area = detect_plot_area_robust(np.array(image))
    t_detect = time.perf_counter() - t0

    t1 = time.perf_counter()
    skeleton, binary, adj_area = preprocess_bw(image, plot_area, config=config)
    t_preprocess = time.perf_counter() - t1

    t2 = time.perf_counter()
    curves = extract_bw_curves(
        image, num_curves, plot_area, config=config,
    )
    t_extract = time.perf_counter() - t2

    summary = {
        "image": str(img_path),
        "plot_area": list(plot_area),
        "skeleton_shape": list(skeleton.shape),
        "num_curves_found": len(curves),
        "curve_lengths": {str(k): len(v) for k, v in curves.items()},
        "time_detect_s": round(t_detect, 3),
        "time_preprocess_s": round(t_preprocess, 3),
        "time_extract_s": round(t_extract, 3),
    }

    # Optionally compute BW confidence metrics
    try:
        from core.metrics import compute_bw_confidence

        # Convert curves to local coords for confidence computation
        p_left, p_top = plot_area[0], plot_area[1]
        local_curves = {
            k: [(x - p_left, y - p_top) for x, y in v]
            for k, v in curves.items()
        }
        conf = compute_bw_confidence(skeleton, binary, local_curves)
        summary["bw_confidence"] = conf
    except Exception as exc:
        logger.warning("confidence computation failed: %s", exc)

    # Save summary
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem
    with open(out_dir / f"{stem}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save skeleton preview
    skel_img = Image.fromarray((skeleton.astype(np.uint8) * 255))
    skel_img.save(out_dir / f"{stem}_skeleton.png")

    logger.info(
        "  -> %d curves, detect=%.3fs preprocess=%.3fs extract=%.3fs",
        len(curves), t_detect, t_preprocess, t_extract,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", type=Path, default=None,
        help="Single image to test (default: all under input/)",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=_ROOT / "input",
        help="Directory to scan for images (default: input/)",
    )
    parser.add_argument(
        "--num-curves", type=int, default=2,
        help="Expected number of curves (default: 2)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug overlays (saved to output dir)",
    )
    parser.add_argument(
        "--no-graph", action="store_true",
        help="Disable graph-based extraction (use legacy pipeline only)",
    )
    parser.add_argument(
        "--no-clahe", action="store_true",
        help="Disable CLAHE preprocessing",
    )
    args = parser.parse_args()

    job_id = time.strftime("%Y%m%d_%H%M%S")
    out_root = _ROOT / "output" / job_id / "debug_bw"
    out_root.mkdir(parents=True, exist_ok=True)

    # Build config from flags
    cfg = BWPipelineConfig(
        use_graph_extraction=not args.no_graph,
        clahe_clip_limit=0.0 if args.no_clahe else DEFAULT_CONFIG.clahe_clip_limit,
        debug_bw=args.debug,
        debug_bw_dir=str(out_root) if args.debug else "",
    )

    if args.image:
        images = [args.image]
    else:
        images = _collect_images(args.input_dir)

    if not images:
        logger.error("No images found")
        sys.exit(1)

    logger.info("Testing %d image(s), output → %s", len(images), out_root)
    summaries = []
    for img_path in images:
        try:
            s = _run_one(img_path, out_root, cfg, num_curves=args.num_curves)
            summaries.append(s)
        except Exception as exc:
            logger.error("FAILED on %s: %s", img_path, exc, exc_info=True)
            summaries.append({"image": str(img_path), "error": str(exc)})

    # Write aggregate report
    report_path = out_root / "report.json"
    with open(report_path, "w") as f:
        json.dump(summaries, f, indent=2)
    logger.info("Report saved to %s", report_path)

    # Print summary table
    print(f"\n{'Image':<50} {'Curves':>6} {'Time(s)':>8}")
    print("-" * 66)
    for s in summaries:
        name = Path(s["image"]).name if "image" in s else "?"
        nc = s.get("num_curves_found", "ERR")
        tt = sum(s.get(k, 0) for k in
                 ("time_detect_s", "time_preprocess_s", "time_extract_s"))
        print(f"{name:<50} {nc:>6} {tt:>8.3f}")


if __name__ == "__main__":
    main()
