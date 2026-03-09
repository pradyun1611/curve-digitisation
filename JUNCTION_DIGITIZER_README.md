# Junction-Aware Multi-Curve Digitizer

## The Problem: Why Plain A* Fails at Junctions

When multiple performance curves (e.g. 70%, 80%, 90%, 100%, 110% speed lines) overlap near their peaks, the standard pipeline fails:

```
1. Threshold → binary image
2. Skeletonize → 1px centerlines
3. A* pathfind between anchors → WRONG BRANCH at junctions!
```

**Why?** At an overlap/junction, the skeleton merges into a single node with 3+ connections. Plain A* picks the shortest/nearest branch, which is often the *wrong* curve continuation. There's no directional memory — the algorithm doesn't know which curve it was following.

```
Before junction:     At junction:          After junction (A* error):
                                           
  Curve A ──────╲                          Curve A ──────╲    ╱── Curve B (WRONG!)
                 ╳── junction node ──►                    ╳──╱
  Curve B ──────╱                          Curve B ──────╱    ╲── Curve A (should be here)
```

## The Solution: Tangent-Based Junction Disambiguation + Stateful Dijkstra

This module solves the junction problem with three key innovations:

### 1. Compressed Skeleton Graph
Instead of pixel-level pathfinding, we build a **compressed graph**:
- **Nodes** = skeleton endpoints (degree 1) + junction pixels (degree ≥ 3)
- **Edges** = pixel chains between nodes, storing full geometry

This reduces a 50,000-pixel skeleton to ~50 nodes + edges.

### 2. Junction Disambiguation via PCA Tangent Pairing
At each junction node, we:
1. Compute the **tangent direction** of each incident edge using PCA on the 12 pixels nearest to the junction.
2. Score all edge pairs by **continuation quality**: cosine similarity of anti-parallel tangents (smooth continuation = tangents point in opposite directions).
3. **Greedily pair** edges: best continuation pair first, then remaining.

This tells us: "If you arrive at this junction via edge E1, you should continue on edge E3" (not E2 or E4).

### 3. Stateful Dijkstra with Direction Memory
The pathfinder's state is `(node, incoming_edge_id)` — not just `node`. This means:
- At a junction, the cost depends on **which edge you came from**
- **Turn penalty**: `λ × angle²` — sharp turns are expensive
- **Junction pairing bonus**: if the (incoming, outgoing) edge pair matches a PCA pairing, the turn cost is discounted
- **Curve priors**: x-monotonicity penalty (curves go left→right), curvature penalty, darkness penalty (prefer dark strokes)

### 4. Beam Search for Multi-Curve Extraction
Without user anchors, the system:
1. Seeds from leftmost endpoints to rightmost endpoints
2. For each pair, runs **K=5 diverse paths** (progressively forbidding junction-adjacent edges from previous results)
3. Greedily selects non-overlapping curves by cost/span ratio

## Architecture

```
Image → deskew_image() → preprocess() → build_graph() → disambiguate_junctions()
                              │                │                    │
                              ▼                ▼                    ▼
                          skeleton         GraphEdges          JunctionPairings
                          binary           endpoints               │
                          gray             junctions               │
                              │                │                    │
                              ▼                ▼                    ▼
                      ┌───────────────────────────────────────────────┐
                      │ trace_curve_with_anchors()  (if anchors)      │
                      │         OR                                    │
                      │ auto_trace_curves()  (beam search)            │
                      │         using stateful_dijkstra()             │
                      └───────────────────────────────────────────────┘
                              │
                              ▼
                      postprocess_curve() → spline fit → resample
                              │
                              ▼
                      map_pixels_to_data() → calibrated coordinates
                              │
                              ▼
                      export_curves_json/csv + QA overlay + debug view
```

## Usage

### CLI
```bash
# Basic (auto-detect curves, no calibration)
python digitize.py --image input/scan.png --out output_dir

# With calibration and anchors
python digitize.py --image input/scan.png \
    --calib calib.json --anchors anchors.json --out output_dir

# Specify number of curves, enable debug
python digitize.py --image input/scan.png --num-curves 5 --debug -v

# Tune parameters
python digitize.py --image input/scan.png --beam-width 8 --lambda-turn 12.0
```

### Python API
```python
from core.junction_digitizer import digitize, JunctionConfig

config = JunctionConfig(
    beam_width=5,
    lambda_turn=8.0,
    debug=True,
)

result = digitize(
    image_path="input/expander_efficiency.png",
    calib_path="calib.json",    # optional
    anchors_path="anchors.json", # optional
    output_dir="output/",
    config=config,
    num_curves=5,
)

# result["curves"]       → {label: [(x,y), ...]} in data coords
# result["curves_pixel"] → same in pixel coords
# result["summary"]      → plain English summary
# result["files_written"] → list of output files
```

### Calibration JSON Formats

**3-point affine (recommended):**
```json
{
  "mode": "3point",
  "points": [
    {"pixel": [100, 400], "data": [0, 40]},
    {"pixel": [500, 400], "data": [200000, 40]},
    {"pixel": [100, 50],  "data": [0, 70]}
  ]
}
```

**2-per-axis (after deskew):**
```json
{
  "mode": "2axis",
  "x_refs": [{"pixel": 100, "value": 0}, {"pixel": 500, "value": 200000}],
  "y_refs": [{"pixel": 400, "value": 40}, {"pixel": 50, "value": 70}],
  "plot_area": [100, 50, 500, 400]
}
```

### Anchors JSON Format
```json
{
  "curves": [
    {"label": "70%",  "anchors": [[50, 380], [200, 300], [350, 380]]},
    {"label": "80%",  "anchors": [[80, 350], [250, 250], [400, 350]]},
    {"label": "100%", "anchors": [[120, 300], [300, 200], [450, 300]]}
  ]
}
```

## Output Files

| File | Content |
|------|---------|
| `curves.json` | Per-curve (x,y) data points |
| `curves.csv` | Tabular export, one column pair per curve |
| `qa_overlay.png` | Original image with colored extracted curves |
| `debug_junctions.png` | Skeleton graph with junction nodes, pairings, and chosen branches |

## Cost Function Details

The stateful Dijkstra cost for traversing edge `e` from node `n` arriving via edge `e_prev`:

```
cost = length(e)                                    # base: edge pixel count
     + μ × curvature(e)                             # penalize wiggly edges
     + ν × mean_darkness(e)/255 × length(e)         # prefer dark strokes
     + penalty_if(Δx < 0) × 50 × |Δx|              # x must go left→right
     + λ × turn_angle(e_prev→e)²                    # smooth turns at junctions
       × discount_if_paired(e_prev, e)              # bonus for PCA-paired edges
```

Default weights: `λ=8.0`, `μ=2.0`, `ν=0.5`.

## Testing

```bash
# Run all junction digitizer tests (22 tests)
pytest tests/test_junction_digitizer.py -v

# Generate synthetic test images for visual inspection
python tests/test_junction_digitizer.py --generate tests/synthetic
```

### Synthetic Test Cases

| Image | Purpose |
|-------|---------|
| `overlapping_arcs_5.png` | 5 parabolic arcs overlapping near peaks |
| `crossing_curves.png` | X-pattern crossing (definite junction) |
| `near_parallel_3.png` | 3 curves merging at one point |
| `thick_overlapping_4.png` | Thick strokes (harder skeleton) |
| `noisy_arcs_3.png` | Gaussian noise on curve positions |

## Dependencies

- `opencv-python` (cv2) — image I/O, morphology, Hough
- `numpy` — array operations
- `scipy` — spline interpolation, distance transforms
- `scikit-image` — skeletonize fallback (if cv2.ximgproc unavailable)
- `pytesseract` (optional) — OCR for image summary
