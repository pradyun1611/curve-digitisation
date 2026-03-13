#!/usr/bin/env python
"""
Batch test for all 11 grayscale images in input/distorted/grayscale.

Tests the BW extraction pipeline directly (without OpenAI) using
hardcoded axis info derived from the chart labels visible in each image.
Generates overlay images and coordinate CSV for manual inspection.
"""
import sys, pathlib, json, logging
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("test_all_grayscale")

from PIL import Image
import numpy as np

from core.image_processor import CurveDigitizer
from core.bw_pipeline import extract_bw_curves, DEFAULT_CONFIG
from core.bw_fit import fit_bw_curve

INPUT_DIR = pathlib.Path("input/distorted/grayscale")
OUTPUT_DIR = pathlib.Path("output/test_grayscale_batch")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Axis info for each image (derived from chart labels)
# Order matches sorted filenames in the directory
AXIS_INFO = [
    # Img 1  - Screenshot 2026-03-10 115627.png  - Shaft Power vs Suction Capacity
    {"xMin": 10000, "xMax": 34000, "yMin": 0, "yMax": 3500,
     "xUnit": "M3/HR", "yUnit": "KW", "num_curves": 5},
    # Img 2  - Screenshot 2026-03-11 160039.png  - Dual: Discharge Pressure + Shaft Power
    {"xMin": 10000, "xMax": 34000, "yMin": 0, "yMax": 3500,
     "xUnit": "M3/HR", "yUnit": "KW", "num_curves": 5},
    # Img 3  - Screenshot 2026-03-12 092027.png  - Isentropic Effy vs Mass Flow
    {"xMin": 0, "xMax": 800000, "yMin": 30, "yMax": 100,
     "xUnit": "KG/HR", "yUnit": "%", "num_curves": 6},
    # Img 4  - Screenshot 2026-03-12 092134.png  - Power vs Mass Flow
    {"xMin": 0, "xMax": 800000, "yMin": 0, "yMax": 10000,
     "xUnit": "KG/HR", "yUnit": "kW", "num_curves": 5},
    # Img 5  - Screenshot 2026-03-12 092228.png  - Polytropic Head vs Volume Flow
    {"xMin": 5000, "xMax": 30000, "yMin": 0, "yMax": 60,
     "xUnit": "M^3/HR", "yUnit": "KJ/KG", "num_curves": 6},
    # Img 6  - Screenshot 2026-03-12 092303.png  - Power vs Mass Flow (surge)
    {"xMin": 100000, "xMax": 700000, "yMin": 0, "yMax": 10000,
     "xUnit": "KG/HR", "yUnit": "kW", "num_curves": 5},
    # Img 7  - Screenshot 2026-03-12 092431.png  - Isentropic Effy vs Mass Flow
    {"xMin": 0, "xMax": 800000, "yMin": 30, "yMax": 100,
     "xUnit": "KG/HR", "yUnit": "%", "num_curves": 7},
    # Img 8  - Screenshot 2026-03-12 092508.png  - Power vs Mass Flow (linear)
    {"xMin": 0, "xMax": 800000, "yMin": 0, "yMax": 8000,
     "xUnit": "KG/HR", "yUnit": "kW", "num_curves": 5},
    # Img 9  - Screenshot 2026-03-12 092548.png  - Polytropic Head vs Volume Flow (surge)
    {"xMin": 5000, "xMax": 30000, "yMin": 0, "yMax": 60,
     "xUnit": "M^3/HR", "yUnit": "KJ/KG", "num_curves": 7},
    # Img 10 - Screenshot 2026-03-12 092622.png  - Power vs Mass Flow (surge, curved)
    {"xMin": 100000, "xMax": 700000, "yMin": 0, "yMax": 10000,
     "xUnit": "KG/HR", "yUnit": "kW", "num_curves": 6},
    # Img 11 - Screenshot 2026-03-12 092701.png  (if exists)
    {"xMin": 100000, "xMax": 700000, "yMin": 0, "yMax": 10000,
     "xUnit": "KG/HR", "yUnit": "kW", "num_curves": 6},
]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_test():
    images = sorted(INPUT_DIR.glob("*.png"))
    if not images:
        logger.error("No images found in %s", INPUT_DIR)
        return

    logger.info("Found %d images", len(images))
    summary = []

    for img_idx, img_path in enumerate(images):
        if img_idx >= len(AXIS_INFO):
            break

        info = AXIS_INFO[img_idx]
        num_curves = info.pop("num_curves", 5)
        img_num = img_idx + 1
        logger.info("=" * 60)
        logger.info("Image %d: %s  (expecting %d curves)", img_num, img_path.name, num_curves)

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Detect plot area
        digitizer = CurveDigitizer(info)
        plot_area = digitizer.detect_plot_area(img)
        plot_area = digitizer._refine_plot_area_with_ticks(img, plot_area)
        logger.info("  Plot area: %s  (image %dx%d)", plot_area, w, h)

        # Extract curves
        curves = extract_bw_curves(
            img, num_curves, plot_area,
            ignore_dashed=True,
            extend_ends=True,
        )
        logger.info("  Extracted %d curves", len(curves))

        # Fit each curve and compute coordinates
        img_out_dir = OUTPUT_DIR / f"img_{img_num:02d}"
        img_out_dir.mkdir(parents=True, exist_ok=True)

        all_fitted = {}
        for cid, pixels in sorted(curves.items()):
            axis_coords = digitizer.normalize_to_axis(pixels, w, h, plot_area)
            fit_result, cleaned = fit_bw_curve(axis_coords)

            # Clamp fitted y-values to axis range
            for p in fit_result.get("fitted_points", []):
                p["y"] = max(info["yMin"], min(info["yMax"], p["y"]))

            fitted_pts = fit_result.get("fitted_points", [])
            xs_px = [p[0] for p in pixels]
            ys_px = [p[1] for p in pixels]

            logger.info(
                "  Curve %d: %d pts, px_x=[%d-%d], fit_method=%s, r2=%.4f, deg=%s",
                cid, len(pixels),
                min(xs_px) if xs_px else 0, max(xs_px) if xs_px else 0,
                fit_result.get("fit_method", "?"),
                fit_result.get("r_squared", 0),
                fit_result.get("degree", "?"),
            )

            # Save coordinates
            if fitted_pts:
                all_fitted[cid] = fitted_pts
                csv_path = img_out_dir / f"curve_{cid}.csv"
                with open(csv_path, "w") as f:
                    f.write(f"x ({info.get('xUnit','x')}),y ({info.get('yUnit','y')})\n")
                    for p in fitted_pts:
                        f.write(f"{p['x']:.4f},{p['y']:.4f}\n")

        # Generate overlay image
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(np.array(img))
        colors = ['red', 'lime', 'blue', 'orange', 'magenta', 'cyan', 'yellow']
        for cid, pixels in sorted(curves.items()):
            c = colors[cid % len(colors)]
            xs = [p[0] for p in pixels]
            ys = [p[1] for p in pixels]
            ax.plot(xs, ys, '.', color=c, markersize=1, alpha=0.5)
        ax.set_title(f"Image {img_num}: {img_path.name}")
        ax.axis("off")
        fig.savefig(str(img_out_dir / "overlay.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Generate fitted curve plot (in data coordinates)
        if all_fitted:
            fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
            ax2.set_xlim(info["xMin"], info["xMax"])
            ax2.set_ylim(info["yMin"], info["yMax"])
            for cid, fps in sorted(all_fitted.items()):
                c = colors[cid % len(colors)]
                ax2.plot([p["x"] for p in fps], [p["y"] for p in fps],
                         color=c, linewidth=1.5, label=f"Curve {cid}")
            ax2.legend(fontsize=8)
            ax2.set_title(f"Image {img_num}: Fitted curves")
            ax2.set_xlabel(info.get("xUnit", "x"))
            ax2.set_ylabel(info.get("yUnit", "y"))
            ax2.grid(True, alpha=0.3)
            fig2.savefig(str(img_out_dir / "fitted.png"), dpi=150, bbox_inches="tight")
            plt.close(fig2)

        # Coordinate range summary
        for cid, fps in sorted(all_fitted.items()):
            if fps:
                x_vals = [p["x"] for p in fps]
                y_vals = [p["y"] for p in fps]
                logger.info(
                    "  C%d: x=[%.0f,%.0f] y=[%.0f,%.0f]",
                    cid, min(x_vals), max(x_vals), min(y_vals), max(y_vals),
                )

        summary.append({
            "image": img_num,
            "file": img_path.name,
            "num_curves_extracted": len(curves),
            "num_curves_expected": num_curves,
            "plot_area": list(plot_area),
        })

    # Save summary
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("Batch test complete. Results in: %s", OUTPUT_DIR)
    for s in summary:
        status = "OK" if s["num_curves_extracted"] >= s["num_curves_expected"] else "LOW"
        logger.info(
            "  Img %2d: %s curves (expected %d) [%s]",
            s["image"], s["num_curves_extracted"], s["num_curves_expected"], status
        )


if __name__ == "__main__":
    run_test()
