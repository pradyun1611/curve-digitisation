#!/usr/bin/env python
"""Generate synthetic test images for curve digitization tests.

Creates:
  tests/data/input_color.png  – 400x300 white image with 5 colored curves
  tests/data/input_bw.png     – 400x300 white image with 3 grayscale curves
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
from PIL import Image, ImageDraw

W, H = 400, 300
DATA_DIR = _root / "tests" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _draw_curve(draw: ImageDraw.ImageDraw, xs: np.ndarray, ys: np.ndarray,
                color: tuple, width: int = 2) -> None:
    pts = list(zip(xs.astype(int).tolist(), ys.astype(int).tolist()))
    draw.line(pts, fill=color, width=width)


def make_color_image() -> Path:
    """5 colored curves on a white background with black axes."""
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw axes
    draw.line([(40, 10), (40, H - 30)], fill=(0, 0, 0), width=2)
    draw.line([(40, H - 30), (W - 10, H - 30)], fill=(0, 0, 0), width=2)

    xs = np.linspace(45, W - 15, 200)

    # Curve 1: red – gentle parabola
    ys_red = 50 + 0.0005 * (xs - 200) ** 2
    _draw_curve(draw, xs, ys_red, (255, 0, 0))

    # Curve 2: blue – sine wave
    ys_blue = 150 + 40 * np.sin((xs - 45) / 50)
    _draw_curve(draw, xs, ys_blue, (0, 0, 255))

    # Curve 3: green – decreasing line
    ys_green = 250 - 0.5 * (xs - 45)
    _draw_curve(draw, xs, ys_green, (0, 180, 0))

    # Curve 4: orange – rising
    ys_orange = 80 + 0.3 * (xs - 45)
    _draw_curve(draw, xs, ys_orange, (255, 140, 0))

    # Curve 5: purple – shallow U
    ys_purple = 220 + 0.0003 * (xs - 200) ** 2
    _draw_curve(draw, xs, ys_purple, (150, 0, 200))

    path = DATA_DIR / "input_color.png"
    img.save(str(path))
    print(f"Saved: {path}")
    return path


def make_bw_image() -> Path:
    """3 grayscale curves (all dark gray on white) with black axes."""
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Axes
    draw.line([(40, 10), (40, H - 30)], fill=(0, 0, 0), width=2)
    draw.line([(40, H - 30), (W - 10, H - 30)], fill=(0, 0, 0), width=2)

    xs = np.linspace(45, W - 15, 200)

    # Curve 1: dark gray – upper
    ys1 = 60 + 0.0004 * (xs - 200) ** 2
    _draw_curve(draw, xs, ys1, (60, 60, 60), width=2)

    # Curve 2: medium gray – middle
    ys2 = 140 + 30 * np.sin((xs - 45) / 60)
    _draw_curve(draw, xs, ys2, (90, 90, 90), width=2)

    # Curve 3: dark gray – lower, decreasing
    ys3 = 240 - 0.35 * (xs - 45)
    _draw_curve(draw, xs, ys3, (70, 70, 70), width=2)

    path = DATA_DIR / "input_bw.png"
    img.save(str(path))
    print(f"Saved: {path}")
    return path


if __name__ == "__main__":
    make_color_image()
    make_bw_image()
    print("Done.")
