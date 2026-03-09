import json

with open('output/20260309_104033/curve_digitization.json') as f:
    data = json.load(f)

for name, curve in data.get('curves', {}).items():
    pts = curve.get('raw_pixel_points', [])
    print(f"{name}: {len(pts)} points, label={curve['label']}")
    if len(pts) > 0:
        print(f"  Start: {pts[0]}, End: {pts[-1]}")
