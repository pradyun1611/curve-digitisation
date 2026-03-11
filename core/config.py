"""
Pipeline configuration with general defaults.

All thresholds/kernels can be overridden per-run via keyword arguments
in the main pipeline functions.  This module provides a single place to
view and tune the default values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class BWPipelineConfig:
    """Configuration for the B/W curve extraction pipeline.

    Every parameter has a general default that works well across many
    different chart styles.  Override individual values as needed.
    """

    # ── ROI / plot-area detection ──
    roi_dark_threshold: int = 80
    roi_line_ratio: float = 0.25
    roi_padding_frac: float = 0.02

    # ── Gridline removal (morphological) ──
    grid_h_kernel_ratio: float = 0.18      # horizontal kernel = img_w * ratio
    grid_v_kernel_ratio: float = 0.18      # vertical kernel = img_h * ratio
    grid_fill_threshold: float = 0.75      # fallback: row/col fill > threshold
    grid_protect_curves: bool = True       # protect curve crossings
    grid_many_lines_threshold: int = 6     # > N detected lines → lower span ratio
    grid_span_ratio_strict: float = 0.80   # few lines: keep only near-full-span
    grid_span_ratio_loose: float = 0.40    # many lines: also remove partial lines

    # ── Text removal (CC-based) ──
    text_area_max_ratio: float = 0.012     # max component area / plot area
    text_aspect_min: float = 0.15          # aspect ratio bounds
    text_aspect_max: float = 6.0
    text_min_curve_span_ratio: float = 0.12
    text_evidence_threshold: float = 0.35  # min score to classify as text

    # ── Tick mark removal ──
    tick_max_length: int = 20              # max tick length in pixels
    tick_max_thickness: int = 5            # max tick width
    tick_axis_proximity: int = 12          # max px from edge

    # ── Morphological closing (gap bridging) ──
    close_kernel_h: int = 3
    close_kernel_w: int = 9
    close_wide_kernel_w: int = 17          # second pass for heavily cleaned images
    close_wide_fg_threshold: float = 0.08  # trigger wide close when fg < this

    # ── Curve extraction ──
    min_curve_pixels: int = 15
    min_curve_span_ratio: float = 0.08
    dashed_threshold: float = 0.45
    text_threshold: float = 0.50

    # ── Curve exclusion ──
    # Modes: '', 'topmost', 'bottommost', 'steepest', 'thickest', 'longest'
    exclude_curve_mode: str = ""

    # ── Smoothing ──
    smoothing_polyorder: int = 3

    # ── A* tracing ──
    snap_radius: int = 50
    curvature_penalty: float = 2.0
    gap_bridge_max: int = 30
    gap_bridge_cost: float = 4.0
    astar_max_iters: int = 500_000          # hard cap on A* iterations
    anchor_min_success_rate: float = 0.40   # fall back if fewer anchors succeed

    # ── Surge / dashed-line detection ──
    surge_dashed_threshold: float = 0.18
    surge_min_span_ratio: float = 0.08
    surge_gap_periodic_cv: float = 1.2      # coefficient-of-variation cutoff
    surge_min_thickness: float = 2.5
    surge_min_gaps: int = 2

    # ── Endpoint extension ──
    extend_search_radius: int = 20
    extend_cone_angle: float = 45.0
    extend_max_extension: int = 100

    # ── Column-scan tracker (multi-curve) ──
    tracker_max_y_jump_ratio: float = 0.03
    tracker_min_track_width_ratio: float = 0.15
    tracker_max_x_gap_ratio: float = 0.10
    tracker_min_coverage: float = 0.40
    tracker_max_slope: float = 3.0         # max y_span/x_span

    # ── Enhanced preprocessing (blur / noise robustness) ──
    clahe_clip_limit: float = 2.0          # CLAHE clip limit (0 = disabled)
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    use_blackhat: bool = False             # black-hat morphology for stroke enhancement
    blackhat_kernel: Tuple[int, int] = (15, 15)
    adaptive_thresh: bool = True           # fallback to adaptive threshold
    adaptive_block_size: int = 51          # local neighbourhood for adaptive thresh
    adaptive_C: int = 10                   # constant subtracted from local mean

    # ── Hough-based axis / gridline removal ──
    hough_remove_axes: bool = True
    hough_threshold: int = 80
    hough_min_line_length: int = 100
    hough_max_line_gap: int = 10

    # ── Graph-based multi-curve extraction ──
    use_graph_extraction: bool = True      # enable skeleton-graph k-path extraction
    k_paths: int = 5                       # candidate paths per endpoint pair
    max_curves: int = 10                   # hard cap on returned curves
    shared_pixel_penalty: float = 5.0      # penalty for pixel overlap between paths
    junction_penalty: float = 3.0          # extra cost at skeleton junctions
    min_path_length_px: int = 50           # discard paths shorter than this

    # ── Debug ──
    debug_bw: bool = False                 # save intermediate BW debug images
    debug_bw_dir: str = ""                 # directory for BW debug images

    @property
    def close_kernel_size(self) -> Tuple[int, int]:
        return (self.close_kernel_h, self.close_kernel_w)


# Singleton default config
DEFAULT_CONFIG = BWPipelineConfig()
