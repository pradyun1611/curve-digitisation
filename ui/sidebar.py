"""
Sidebar configuration module for Streamlit UI.

Handles sidebar setup, API key configuration, and B/W pipeline controls.
"""

import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def setup_sidebar() -> tuple:
    """
    Setup and configure the sidebar.
    
    Returns:
        Tuple of (api_key, endpoint, deployment_name, output_dir)
    """
    with st.sidebar:
        st.title("ℹ️ About")
        st.write("""
        This chatbot helps you extract and digitize curves from performance chart images.
        
        **Features:**
        - Extract axis information and units
        - Identify and isolate curves by color
        - Fit polynomial curves to extracted data
        - Clean noisy data with RANSAC filtering
        - B/W skeleton-based extraction with dashed-line rejection
        - Anchor-guided curve tracing (start/end points)
        - Export results as JSON/CSV
        
        **How to use:**
        1. Configure Azure OpenAI credentials via environment variables
        2. Type a message or upload an image
        3. For image processing, mention the image in your query
        """)
        
        st.divider()

        # ── Pipeline Controls ──
        st.subheader("⚙️ Pipeline Settings")

        # Mode selector
        mode = st.selectbox(
            "Image Mode",
            options=["Auto", "Colored", "B&W"],
            index=0,
            key="pipeline_mode",
            help="Auto detects whether the image is colored or B/W. "
                 "Override if auto-detection is wrong.",
        )

        # B/W-specific controls (only shown when relevant)
        ignore_dashed = True
        smoothing_strength = 0
        use_skeleton = True
        target_curves = 0
        dashed_threshold = 0.45
        text_threshold = 0.50
        if mode in ("Auto", "B&W"):
            with st.expander("B/W Options", expanded=False):
                ignore_dashed = st.checkbox(
                    "Ignore dashed/dotted lines",
                    value=True,
                    key="ignore_dashed",
                    help="Reject dashed / dotted lines during B/W extraction.",
                )
                smoothing_strength = st.slider(
                    "Smoothing strength",
                    min_value=0, max_value=51, value=0, step=2,
                    key="smoothing_strength",
                    help="Savitzky–Golay window size (0 = auto).",
                )
                use_skeleton = st.checkbox(
                    "Use skeleton pipeline (recommended)",
                    value=True,
                    key="use_skeleton",
                    help="New skeleton-based extraction. Uncheck to use "
                         "legacy column-scan tracker.",
                )
                target_curves = st.number_input(
                    "Target curves (0 = auto-detect)",
                    min_value=0, max_value=20, value=0, step=1,
                    key="target_curves",
                    help="Force a specific number of curves. "
                         "0 = use LLM count + auto-detect.",
                )
            with st.expander("B/W Thresholds", expanded=False):
                st.caption("Tune rejection thresholds (advanced).")
                dashed_threshold = st.slider(
                    "Dashed rejection threshold",
                    min_value=0.0, max_value=1.0, value=0.45, step=0.05,
                    key="dashed_threshold",
                    help="Components with dashed_score > threshold are rejected.",
                )
                text_threshold = st.slider(
                    "Text rejection threshold",
                    min_value=0.0, max_value=1.0, value=0.50, step=0.05,
                    key="text_threshold",
                    help="Components with text_score > threshold are rejected.",
                )

        # Axis calibration
        st.subheader("📐 Axis Calibration")
        with st.expander("Manual Axis Bounds", expanded=False):
            st.caption("Override axis bounds if auto-detection is wrong.")
            cal_x_min = st.number_input("X min", value=None, key="cal_x_min")
            cal_x_max = st.number_input("X max", value=None, key="cal_x_max")
            cal_y_min = st.number_input("Y min", value=None, key="cal_y_min")
            cal_y_max = st.number_input("Y max", value=None, key="cal_y_max")

        with st.expander("Plot Area Override (px)", expanded=False):
            st.caption(
                "Override auto-detected plot pixel bounds. "
                "Leave at 0 for auto-detection."
            )
            pa_left = st.number_input("Plot Left (px)", value=0, min_value=0, key="pa_left")
            pa_top = st.number_input("Plot Top (px)", value=0, min_value=0, key="pa_top")
            pa_right = st.number_input("Plot Right (px)", value=0, min_value=0, key="pa_right")
            pa_bottom = st.number_input("Plot Bottom (px)", value=0, min_value=0, key="pa_bottom")

        # Anchor points — multi-curve support
        st.subheader("📍 Anchor Points")
        with st.expander("Start / End Anchors", expanded=False):
            st.caption(
                "Pixel coords for anchor-guided tracing (B/W). "
                "Add one row per curve."
            )
            n_anchor_rows = st.number_input(
                "Number of anchor pairs",
                min_value=0, max_value=20, value=0, step=1,
                key="n_anchor_rows",
            )
            anchor_pairs = []
            for i in range(int(n_anchor_rows)):
                cols = st.columns(4)
                sx = cols[0].number_input(f"S{i+1} X", value=0, key=f"a_sx_{i}")
                sy = cols[1].number_input(f"S{i+1} Y", value=0, key=f"a_sy_{i}")
                ex = cols[2].number_input(f"E{i+1} X", value=0, key=f"a_ex_{i}")
                ey = cols[3].number_input(f"E{i+1} Y", value=0, key=f"a_ey_{i}")
                if sx or sy or ex or ey:
                    anchor_pairs.append(((sx, sy), (ex, ey)))

            use_anchors = st.checkbox(
                "Use anchors", value=False, key="use_anchors",
            ) if n_anchor_rows > 0 else False

        # Curve exclusion
        with st.expander("Curve Exclusion", expanded=False):
            st.caption(
                "Optionally exclude one extracted curve by heuristic. "
                "Use 'steepest' to remove a surge line."
            )
            exclude_curve_mode = st.selectbox(
                "Exclude curve mode",
                options=["", "steepest", "topmost", "bottommost",
                         "longest", "thickest"],
                index=0,
                key="exclude_curve_mode",
                help="Remove one curve matching this criterion.",
            )

        # Debug toggle
        show_debug = st.checkbox(
            "Show debug overlay",
            value=False,
            key="show_debug",
            help="Display skeleton, chosen path, rejected components, axis bounds.",
        )

        st.divider()

        # Clear chat button
        if st.button("🗑️ Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.session_state.results_history = []
            st.rerun()

    # Store pipeline settings in session state for access elsewhere
    st.session_state["pipeline_settings"] = {
        "mode": {"Auto": "auto", "Colored": "color", "B&W": "bw"}.get(mode, "auto"),
        "ignore_dashed": ignore_dashed,
        "smoothing_strength": smoothing_strength,
        "use_skeleton": use_skeleton,
        "show_debug": show_debug,
        "target_curves": target_curves,
        "dashed_threshold": dashed_threshold,
        "text_threshold": text_threshold,
        "exclude_curve_mode": exclude_curve_mode if 'exclude_curve_mode' in dir() else "",
        "use_anchors": use_anchors if 'use_anchors' in dir() else False,
        "anchors": anchor_pairs if (use_anchors if 'use_anchors' in dir() else False) and anchor_pairs else None,
        "calibration_overrides": {},
        "plot_area_override": None,
    }

    # Handle plot area override
    if pa_left > 0 or pa_top > 0 or pa_right > 0 or pa_bottom > 0:
        if pa_right > pa_left and pa_bottom > pa_top:
            st.session_state["pipeline_settings"]["plot_area_override"] = (
                pa_left, pa_top, pa_right, pa_bottom,
            )

    # Handle calibration overrides
    cal_override = {}
    if cal_x_min is not None:
        cal_override["xMin"] = cal_x_min
    if cal_x_max is not None:
        cal_override["xMax"] = cal_x_max
    if cal_y_min is not None:
        cal_override["yMin"] = cal_y_min
    if cal_y_max is not None:
        cal_override["yMax"] = cal_y_max
    if cal_override:
        st.session_state["pipeline_settings"]["calibration_overrides"] = cal_override
    
    # Load credentials from environment variables
    api_key = os.getenv('OPENAI_API_KEY', '')
    endpoint = os.getenv('AZURE_ENDPOINT', '')
    deployment_name = os.getenv('AZURE_DEPLOYMENT_NAME', '')
    output_dir = "./output/"
    
    return api_key, endpoint, deployment_name, output_dir
