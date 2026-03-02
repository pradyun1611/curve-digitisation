"""
Image mode router — decides Colored vs B/W processing path.

Uses HSV saturation statistics to classify images.  The colour pipeline
is preserved exactly; B/W images are routed to the new skeleton-based
pipeline.

Supports three modes:
  - ``"auto"``  : detect from image content (default)
  - ``"color"`` : force colour pipeline
  - ``"bw"``    : force B/W pipeline
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Thresholds for auto-detection
_SAT_MEAN_THRESHOLD = 25        # mean saturation below this → likely B/W
_SAT_PIXEL_THRESHOLD = 30       # per-pixel S threshold
_SAT_PIXEL_RATIO = 0.08         # % of non-bg pixels that must be colourful
_PALETTE_SIZE_THRESHOLD = 8     # unique hue bins; ≤ this → B/W


def classify_image_mode(
    image: Image.Image,
    *,
    mode_override: str = "auto",
) -> str:
    """Classify an image as ``"color"`` or ``"bw"``.

    Parameters
    ----------
    image : PIL.Image
        Input image (will be converted to RGB internally).
    mode_override : str
        ``"auto"`` (detect), ``"color"`` (force), ``"bw"`` (force).

    Returns
    -------
    str
        ``"color"`` or ``"bw"``.
    """
    if mode_override == "color":
        logger.info("router: mode forced to 'color'")
        return "color"
    if mode_override in ("bw", "grayscale"):
        logger.info("router: mode forced to 'bw'")
        return "bw"

    # Auto-detect using HSV saturation
    img_rgb = np.array(image.convert("RGB"))
    result, details = _compute_saturation_stats(img_rgb)

    logger.info(
        "router: auto-detect → %s  (S_mean=%.1f, colorful_ratio=%.3f, "
        "hue_bins=%d)",
        result, details["s_mean"], details["colorful_ratio"],
        details["hue_bins"],
    )
    return result


def _compute_saturation_stats(
    img_rgb: np.ndarray,
) -> Tuple[str, dict]:
    """Compute HSV saturation statistics on non-background pixels.

    Returns (mode, details_dict).
    """
    # Convert to HSV via PIL (H: 0-179, S: 0-255, V: 0-255)
    hsv = np.array(Image.fromarray(img_rgb).convert("HSV"))

    h_channel = hsv[:, :, 0].astype(np.float32)
    s_channel = hsv[:, :, 1].astype(np.float32)
    v_channel = hsv[:, :, 2].astype(np.float32)

    # Exclude background: very bright (white) or very dark (black)
    non_bg = (v_channel > 30) & (v_channel < 240)
    n_non_bg = int(non_bg.sum())

    if n_non_bg < 100:
        return "bw", {"s_mean": 0, "colorful_ratio": 0, "hue_bins": 0}

    s_vals = s_channel[non_bg]
    h_vals = h_channel[non_bg]

    s_mean = float(np.mean(s_vals))
    colorful_mask = s_vals > _SAT_PIXEL_THRESHOLD
    colorful_ratio = float(np.sum(colorful_mask)) / n_non_bg

    # Count distinct hue bins (18 bins of 10° each, only for saturated pixels)
    if np.sum(colorful_mask) > 10:
        h_colorful = h_vals[colorful_mask]
        hue_hist, _ = np.histogram(h_colorful, bins=18, range=(0, 180))
        hue_bins = int(np.sum(hue_hist > 0))
    else:
        hue_bins = 0

    details = {
        "s_mean": s_mean,
        "colorful_ratio": colorful_ratio,
        "hue_bins": hue_bins,
    }

    # Decision logic
    if s_mean < _SAT_MEAN_THRESHOLD and colorful_ratio < _SAT_PIXEL_RATIO:
        return "bw", details
    if s_mean < _SAT_MEAN_THRESHOLD and hue_bins <= _PALETTE_SIZE_THRESHOLD:
        return "bw", details

    return "color", details
