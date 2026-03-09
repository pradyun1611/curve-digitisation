"""
Click-to-place anchor canvas for interactive curve tracing.

Uses streamlit-image-coordinates to let users click MULTIPLE waypoints
along a curve.  The pipeline traces through all waypoints in order,
producing exactly one curve per set of waypoints.
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Optional, Tuple

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:
    streamlit_image_coordinates = None  # graceful degradation


# ── Constants ──
_MARKER_RADIUS = 5
_MAX_DISPLAY_WIDTH = 800
_COLORS = [
    (0, 200, 0), (220, 40, 40), (40, 100, 255), (255, 165, 0),
    (180, 0, 180), (0, 200, 200), (128, 0, 0), (0, 128, 128),
]


def _init_state() -> None:
    """Ensure all session-state keys exist."""
    if "click_curves" not in st.session_state:
        # list of {"waypoints": [(x,y), ...]}
        st.session_state.click_curves = []
    if "picking" not in st.session_state:
        st.session_state.picking = None  # e.g. "add_0"
    if "last_click_xy" not in st.session_state:
        st.session_state.last_click_xy = None


def render_anchor_canvas(image_bytes: bytes) -> Optional[List[List[Tuple[int, int]]]]:
    """Render the interactive click canvas with multi-waypoint support.

    Click many points along a curve — the tracer follows them in order.

    Returns
    -------
    list[list[(x, y)]] or None
        List of waypoint lists per curve (only curves with >= 2 waypoints).
    """
    if streamlit_image_coordinates is None:
        st.warning("Install `streamlit-image-coordinates` to enable click-to-place anchors.")
        return None

    _init_state()

    img = Image.open(__import__("io").BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = img.size

    display_w = min(orig_w, _MAX_DISPLAY_WIDTH)
    scale = orig_w / display_w

    display_img = img.resize((display_w, int(orig_h / scale)), Image.LANCZOS)
    display_img_marked = _draw_markers_display(display_img, scale)

    # Picking indicator
    picking = st.session_state.picking
    if picking and picking.startswith("add_"):
        curve_idx = int(picking.split("_")[1]) + 1
        n_pts = len(st.session_state.click_curves[int(picking.split("_")[1])].get("waypoints", []))
        st.info(
            f"🎯 **Click on the image** to add points for Curve {curve_idx} "
            f"({n_pts} placed). Click along the curve — more points = better trace!"
        )

    # Clickable image
    coords = streamlit_image_coordinates(
        display_img_marked,
        key="anchor_canvas",
        cursor="crosshair",
    )

    # Handle click — add waypoint to the active curve
    if coords is not None and picking and picking.startswith("add_"):
        click_x = int(coords["x"] * scale)
        click_y = int(coords["y"] * scale)
        if (click_x, click_y) != st.session_state.last_click_xy:
            st.session_state.last_click_xy = (click_x, click_y)
            curve_idx = int(picking.split("_")[1])
            if curve_idx < len(st.session_state.click_curves):
                st.session_state.click_curves[curve_idx].setdefault("waypoints", []).append(
                    (click_x, click_y)
                )
            st.rerun()

    # Curve cards
    _render_curve_cards()

    # Build completed waypoint lists (>= 2 points)
    completed: List[List[Tuple[int, int]]] = []
    for curve in st.session_state.click_curves:
        wps = curve.get("waypoints", [])
        if len(wps) >= 2:
            completed.append(list(wps))

    # Push into pipeline_settings as multi-waypoint anchors
    if completed:
        settings = st.session_state.get("pipeline_settings", {})
        settings["anchors"] = completed
        settings["use_anchors"] = True
        st.session_state["pipeline_settings"] = settings

    return completed if completed else None


def _draw_markers_display(display_img: Image.Image, scale: float) -> Image.Image:
    """Draw waypoint markers and connecting lines at display-space coordinates."""
    overlay = display_img.copy()
    draw = ImageDraw.Draw(overlay)
    r = _MARKER_RADIUS

    for i, curve in enumerate(st.session_state.click_curves):
        col = _COLORS[i % len(_COLORS)]
        wps = curve.get("waypoints", [])
        # Draw lines between consecutive waypoints
        for j in range(len(wps) - 1):
            x1 = int(wps[j][0] / scale)
            y1 = int(wps[j][1] / scale)
            x2 = int(wps[j + 1][0] / scale)
            y2 = int(wps[j + 1][1] / scale)
            draw.line([(x1, y1), (x2, y2)], fill=col, width=1)
        # Draw waypoint dots
        for j, (wx, wy) in enumerate(wps):
            dx = int(wx / scale)
            dy = int(wy / scale)
            draw.ellipse([dx - r, dy - r, dx + r, dy + r],
                         fill=col, outline="white", width=2)
            draw.text((dx + r + 2, dy - r - 2), f"{j + 1}", fill=col)

    return overlay


def _render_curve_cards() -> None:
    """Render per-curve waypoint controls."""

    if st.button("➕ Add Curve", key="add_anchor_curve"):
        st.session_state.click_curves.append({"waypoints": []})
        # Auto-start picking for the new curve
        st.session_state.picking = f"add_{len(st.session_state.click_curves) - 1}"
        st.rerun()

    if not st.session_state.click_curves:
        st.caption(
            "Click **Add Curve** then click points along the curve on the image. "
            "The more points you click, the better the trace."
        )
        return

    for i, curve in enumerate(st.session_state.click_curves):
        wps = curve.get("waypoints", [])
        is_picking = st.session_state.picking == f"add_{i}"

        cols = st.columns([3, 1, 1, 0.5])

        # Info
        n = len(wps)
        if n == 0:
            cols[0].warning(f"Curve {i + 1}: no points")
        elif n == 1:
            cols[0].info(f"Curve {i + 1}: {n} point (need ≥ 2)")
        else:
            cols[0].success(f"Curve {i + 1}: {n} points ✓")

        # Pick / Stop button
        if is_picking:
            if cols[1].button("⏹ Stop", key=f"stop_{i}"):
                st.session_state.picking = None
                st.rerun()
        else:
            if cols[1].button("📌 Pick", key=f"pick_{i}"):
                st.session_state.picking = f"add_{i}"
                st.rerun()

        # Undo last point
        if cols[2].button("↩ Undo", key=f"undo_{i}"):
            if wps:
                wps.pop()
            st.rerun()

        # Delete curve
        if cols[3].button("🗑️", key=f"del_c_{i}"):
            st.session_state.click_curves.pop(i)
            if st.session_state.picking == f"add_{i}":
                st.session_state.picking = None
            st.rerun()
