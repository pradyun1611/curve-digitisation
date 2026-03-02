"""
Click-to-place anchor canvas for interactive curve tracing.

Uses streamlit-image-coordinates to let users click on the uploaded image
and automatically fill start/end anchor coordinates for each curve.
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional, Tuple

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:
    streamlit_image_coordinates = None  # graceful degradation


# ── Constants ──
_MARKER_RADIUS = 6
_MAX_DISPLAY_WIDTH = 800
_START_COLOR = (0, 200, 0)      # green
_END_COLOR = (220, 40, 40)      # red
_FONT_SIZE = 12


def _init_state() -> None:
    """Ensure all session-state keys exist."""
    if "click_curves" not in st.session_state:
        st.session_state.click_curves = []  # list of {"start": (x,y)|None, "end": (x,y)|None}
    if "picking" not in st.session_state:
        st.session_state.picking = None  # e.g. "start_0", "end_1"
    if "last_click_xy" not in st.session_state:
        st.session_state.last_click_xy = None


def _draw_markers(img: Image.Image) -> Image.Image:
    """Draw start/end markers on a copy of *img*."""
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    r = _MARKER_RADIUS

    for i, curve in enumerate(st.session_state.click_curves):
        if curve.get("start"):
            sx, sy = curve["start"]
            draw.ellipse([sx - r, sy - r, sx + r, sy + r],
                         fill=_START_COLOR, outline="white", width=2)
            draw.text((sx + r + 3, sy - r), f"S{i + 1}", fill=_START_COLOR)
        if curve.get("end"):
            ex, ey = curve["end"]
            draw.ellipse([ex - r, ey - r, ex + r, ey + r],
                         fill=_END_COLOR, outline="white", width=2)
            draw.text((ex + r + 3, ey - r), f"E{i + 1}", fill=_END_COLOR)

    return overlay


def render_anchor_canvas(image_bytes: bytes) -> Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """Render the interactive click canvas and per-curve controls.

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of the uploaded image file.

    Returns
    -------
    list[((sx, sy), (ex, ey))] or None
        Completed anchor pairs (only pairs with both start and end).
    """
    if streamlit_image_coordinates is None:
        st.warning("Install `streamlit-image-coordinates` to enable click-to-place anchors.")
        return None

    _init_state()

    img = Image.open(__import__("io").BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = img.size

    # Compute display width (cap at _MAX_DISPLAY_WIDTH)
    display_w = min(orig_w, _MAX_DISPLAY_WIDTH)
    scale = orig_w / display_w  # >1 when image is downscaled

    # Draw existing markers at *display* coordinates
    display_img = img.resize((display_w, int(orig_h / scale)), Image.LANCZOS)
    display_img_marked = _draw_markers_display(display_img, scale)

    # --- picking indicator ---
    picking = st.session_state.picking
    if picking:
        parts = picking.split("_")
        point_type = parts[0].title()
        curve_idx = int(parts[1]) + 1
        st.info(f"🎯 Click on the image to place **{point_type}** point for Curve {curve_idx}")

    # --- clickable image ---
    coords = streamlit_image_coordinates(
        display_img_marked,
        key="anchor_canvas",
        cursor="crosshair",
    )

    # --- handle click ---
    if coords is not None and picking:
        click_x = int(coords["x"] * scale)
        click_y = int(coords["y"] * scale)
        # Deduplicate: ignore if same as last processed click
        if (click_x, click_y) != st.session_state.last_click_xy:
            st.session_state.last_click_xy = (click_x, click_y)
            parts = picking.split("_")
            point_type = parts[0]
            curve_idx = int(parts[1])
            if curve_idx < len(st.session_state.click_curves):
                st.session_state.click_curves[curve_idx][point_type] = (click_x, click_y)
            st.session_state.picking = None
            st.rerun()

    # --- curve cards ---
    _render_curve_cards()

    # --- build anchor pairs ---
    anchor_pairs = []
    for curve in st.session_state.click_curves:
        if curve.get("start") and curve.get("end"):
            anchor_pairs.append((curve["start"], curve["end"]))

    # Push completed pairs into pipeline_settings automatically
    if anchor_pairs:
        settings = st.session_state.get("pipeline_settings", {})
        settings["anchors"] = anchor_pairs
        settings["use_anchors"] = True
        st.session_state["pipeline_settings"] = settings

    return anchor_pairs if anchor_pairs else None


def _draw_markers_display(display_img: Image.Image, scale: float) -> Image.Image:
    """Draw markers at display-space coordinates."""
    overlay = display_img.copy()
    draw = ImageDraw.Draw(overlay)
    r = _MARKER_RADIUS

    for i, curve in enumerate(st.session_state.click_curves):
        if curve.get("start"):
            sx, sy = int(curve["start"][0] / scale), int(curve["start"][1] / scale)
            draw.ellipse([sx - r, sy - r, sx + r, sy + r],
                         fill=_START_COLOR, outline="white", width=2)
            draw.text((sx + r + 3, sy - r), f"S{i + 1}", fill=_START_COLOR)
        if curve.get("end"):
            ex, ey = int(curve["end"][0] / scale), int(curve["end"][1] / scale)
            draw.ellipse([ex - r, ey - r, ex + r, ey + r],
                         fill=_END_COLOR, outline="white", width=2)
            draw.text((ex + r + 3, ey - r), f"E{i + 1}", fill=_END_COLOR)

    return overlay


def _render_curve_cards() -> None:
    """Render per-curve anchor controls (Add / Pick / Clear / Delete)."""

    # Add Curve button
    if st.button("➕ Add Curve", key="add_anchor_curve"):
        st.session_state.click_curves.append({"start": None, "end": None})
        st.rerun()

    if not st.session_state.click_curves:
        st.caption("Click **Add Curve** then pick start & end points on the image.")
        return

    for i, curve in enumerate(st.session_state.click_curves):
        cols = st.columns([2, 1, 2, 1, 0.5])

        # Start info
        if curve.get("start"):
            cols[0].success(f"S{i+1}: ({curve['start'][0]}, {curve['start'][1]})")
        else:
            cols[0].warning(f"S{i+1}: —")

        # Pick Start
        if cols[1].button("📌 Start", key=f"pick_s_{i}"):
            st.session_state.picking = f"start_{i}"
            st.rerun()

        # End info
        if curve.get("end"):
            cols[2].success(f"E{i+1}: ({curve['end'][0]}, {curve['end'][1]})")
        else:
            cols[2].warning(f"E{i+1}: —")

        # Pick End
        if cols[3].button("📌 End", key=f"pick_e_{i}"):
            st.session_state.picking = f"end_{i}"
            st.rerun()

        # Delete
        if cols[4].button("🗑️", key=f"del_c_{i}"):
            st.session_state.click_curves.pop(i)
            st.rerun()
