"""Test filtered extraction on both real images."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from core.image_processor import CurveDigitizer

root = Path(__file__).resolve().parent.parent
axis_info = {"xMin": 0, "xMax": 30000, "yMin": 0, "yMax": 3500,
             "xUnit": "kg/hr", "yUnit": "kW"}

# Test both images
for d in sorted(root.glob("output/2026*"), reverse=True)[:5]:
    inp = [p for p in d.iterdir() if p.name.startswith("input_")]
    if not inp:
        continue
    img_path = inp[0]
    print(f"\n{'='*60}")
    print(f"Image: {img_path.name}")
    
    image = Image.open(str(img_path))
    dig = CurveDigitizer(axis_info)
    pa = dig.detect_plot_area(image)
    
    # Simulate: 5 speed lines expected
    clusters = dig.extract_curves_grayscale(image, 5, pa)
    pw = pa[2] - pa[0]
    
    print(f"Tracks found: {len(clusters)} (requested <=5)")
    for idx, pts in sorted(clusters.items()):
        a = np.array(pts)
        xs, ys = a[:,0], a[:,1]
        xspan = int(xs.max()-xs.min())
        yspan = int(ys.max()-ys.min())
        slope = yspan/max(xspan,1)
        ux = len(set(xs.tolist()))
        cov = ux/max(xspan,1)
        print(f"  T{idx}: {len(pts):3d}pts span={xspan}({xspan/pw:.0%}) "
              f"y_span={yspan} slope={slope:.2f} cov={cov:.2f}")
