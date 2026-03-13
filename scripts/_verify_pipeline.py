#!/usr/bin/env python
"""Quick end-to-end pipeline test on a few images."""
import sys, os, json, base64, logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from core.openai_client import OpenAIClient
from core.image_processor import CurveDigitizer

api_key = os.getenv("OPENAI_API_KEY", "")
endpoint = os.getenv("AZURE_ENDPOINT", "")
deployment = os.getenv("AZURE_DEPLOYMENT_NAME", "")
client = OpenAIClient(api_key, endpoint, deployment)

test_images = [
    ROOT / "input" / "clear" / "grayscale" / "Screenshot 2026-02-25 112637.png",
    ROOT / "input" / "distorted" / "grayscale" / "Screenshot 2026-03-12 092027.png",
]

for img_path in test_images:
    if not img_path.exists():
        print(f"SKIP: {img_path.name}")
        continue
    print(f"\n{'='*60}")
    print(f"IMAGE: {img_path.name}")
    print(f"{'='*60}")

    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    axis_info = client.extract_axis_info(b64)
    print(f"Axis: x=[{axis_info.get('xMin')}..{axis_info.get('xMax')}] {axis_info.get('xUnit')}")
    print(f"      y=[{axis_info.get('yMin')}..{axis_info.get('yMax')}] {axis_info.get('yUnit')}")

    digitizer = CurveDigitizer(axis_info)
    print(f"Digitizer bounds: x=[{digitizer.xMin}..{digitizer.xMax}] y=[{digitizer.yMin}..{digitizer.yMax}]")

    out_dir = str(ROOT / "output" / "test_verify")
    os.makedirs(out_dir, exist_ok=True)
    features = client.extract_curve_features(b64)
    results = digitizer.process_curve_image(str(img_path), features, out_dir)

    curves = results.get("curves", {})
    for cname, cdata in list(curves.items())[:3]:
        fit = cdata.get("fit_result", {})
        fitted_pts = fit.get("fitted_points", [])
        if fitted_pts:
            print(f"  Curve {cname}: {len(fitted_pts)} fitted points")
            first3 = [(p["x"], p["y"]) for p in fitted_pts[:3]]
            last3 = [(p["x"], p["y"]) for p in fitted_pts[-3:]]
            print(f"    First 3: {first3}")
            print(f"    Last 3:  {last3}")
        else:
            coords = cdata.get("axis_coords", [])
            if coords:
                print(f"  Curve {cname}: {len(coords)} axis_coords")
                print(f"    First 3: {coords[:3]}")
                print(f"    Last 3:  {coords[-3:]}")
            else:
                err = cdata.get("error", "no data")
                print(f"  Curve {cname}: {err}")

    inst = results.get("instance_dir", "?")
    print(f"Instance dir: {inst}")

print("\nDone!")
