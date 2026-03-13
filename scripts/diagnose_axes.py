#!/usr/bin/env python
"""
Diagnose axis extraction: sends each input image to GPT and prints the
axis values returned. This helps identify whether GPT is misreading axes.
"""
import sys, json, base64, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from core.openai_client import OpenAIClient
from core.image_processor import CurveDigitizer
from PIL import Image


def get_client():
    api_key = os.getenv("OPENAI_API_KEY", "")
    endpoint = os.getenv("AZURE_ENDPOINT", "")
    deployment = os.getenv("AZURE_DEPLOYMENT_NAME", "")
    if not all([api_key, endpoint, deployment]):
        print("ERROR: Azure credentials not set in .env")
        sys.exit(1)
    return OpenAIClient(api_key, endpoint, deployment)


def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def detect_plot_area(img_path: Path):
    """Use CurveDigitizer to detect plot area."""
    dummy_axis = {"xMin": 0, "xMax": 100, "yMin": 0, "yMax": 100}
    dig = CurveDigitizer(dummy_axis)
    img = Image.open(str(img_path)).convert("RGB")
    pa = dig.detect_plot_area(img)
    return pa, img.size


def main():
    client = get_client()

    folders = [
        ROOT / "input" / "clear" / "grayscale",
        ROOT / "input" / "distorted" / "grayscale",
        ROOT / "input" / "distorted" / "colored",
    ]

    results = {}
    for folder in folders:
        if not folder.exists():
            continue
        images = sorted(folder.glob("*.png"))
        if not images:
            continue
        print(f"\n{'='*70}")
        print(f"FOLDER: {folder.relative_to(ROOT)}")
        print(f"{'='*70}")

        for img_path in images:
            print(f"\n--- {img_path.name} ---")
            b64 = encode_image(img_path)

            # Get axis info from GPT
            axis_info = client.extract_axis_info(b64, "Extract axis information from this performance curve chart")
            print(f"  GPT axis_info:")
            print(f"    xMin={axis_info.get('xMin')}  xMax={axis_info.get('xMax')}  xUnit={axis_info.get('xUnit')}")
            print(f"    yMin={axis_info.get('yMin')}  yMax={axis_info.get('yMax')}  yUnit={axis_info.get('yUnit')}")

            # Detect plot area
            pa, (w, h) = detect_plot_area(img_path)
            print(f"  plot_area: left={pa[0]}, top={pa[1]}, right={pa[2]}, bottom={pa[3]}")
            print(f"  image_size: {w}x{h}")
            print(f"  plot_size: {pa[2]-pa[0]}x{pa[3]-pa[1]}")

            results[str(img_path.relative_to(ROOT))] = {
                "axis_info": axis_info,
                "plot_area": list(pa),
                "image_size": [w, h],
            }

    # Save results
    out_path = ROOT / "output" / "axis_diagnosis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
