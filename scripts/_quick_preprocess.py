#!/usr/bin/env python
"""Quick preprocessing diagnostic for all 11 images."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from PIL import Image
import numpy as np
from core.bw_pipeline import preprocess_bw, DEFAULT_CONFIG
from core.image_processor import CurveDigitizer

images = sorted(pathlib.Path('input/distorted/grayscale').glob('*.png'))
for i, p in enumerate(images):
    try:
        img = Image.open(p).convert('RGB')
        d = CurveDigitizer({'xMin': 0, 'xMax': 100, 'yMin': 0, 'yMax': 100})
        pa = d.detect_plot_area(img)
        pa = d._refine_plot_area_with_ticks(img, pa)
        skel, binary, adj = preprocess_bw(img, pa, config=DEFAULT_CONFIG)
        fg = float(binary.sum()) / max(binary.size, 1)
        print(f'Img {i+1:2d}: binary_fg={fg:.4f} skel_px={int(skel.sum())}')
    except Exception as e:
        print(f'Img {i+1:2d}: ERROR {e}')
