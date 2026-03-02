"""
Result display module for Streamlit UI.

Handles visualization of image processing results, including:
- Input/output image comparisons
- Per-curve metrics and coefficients
- Debug overlays (skeleton, axes bounds, rejected components)
- Export (JSON/CSV download buttons)
"""

import streamlit as st
import json
import csv
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def display_image_results(results: Dict[str, Any]) -> None:
    """
    Display image processing results in a nice format.
    
    Args:
        results: Dictionary containing processing results
    """
    if not results:
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Axis Information")
        axis_info = results.get('axis_info', {})
        
        axis_data = {
            'X-axis': f"{axis_info.get('xMin')} to {axis_info.get('xMax')} ({axis_info.get('xUnit', 'units')})",
            'Y-axis': f"{axis_info.get('yMin')} to {axis_info.get('yMax')} ({axis_info.get('yUnit', 'units')})"
        }
        
        for key, value in axis_data.items():
            st.write(f"**{key}:** {value}")
    
    with col2:
        st.subheader("📈 Image Dimensions")
        dims = results.get('image_dimensions', {})
        st.write(f"**Width:** {dims.get('width', 'N/A')} px")
        st.write(f"**Height:** {dims.get('height', 'N/A')} px")
    
    # ── Input vs Output Image Comparison ──
    input_img = results.get('input_image_path', results.get('image_path', ''))
    output_graphs = results.get('output_graphs', {})
    output_img = output_graphs.get('all_curves', '')
    
    if (input_img and Path(input_img).exists()) or (output_img and Path(output_img).exists()):
        st.subheader("🖼️ Input vs Output Comparison")
        img_col1, img_col2 = st.columns(2)
        
        with img_col1:
            if input_img and Path(input_img).exists():
                st.image(input_img, caption="Input Image", use_container_width=True)
            else:
                st.info("Input image not available.")
        
        with img_col2:
            if output_img and Path(output_img).exists():
                st.image(output_img, caption="Digitized Output", use_container_width=True)
            else:
                st.info("Output graph not available.")
    
    # ── Overall Graph Metrics ──
    overall = results.get('overall_metrics', {})
    if overall and overall.get('delta_value') is not None:
        st.subheader("📐 Overall Graph Quality Metrics")
        st.caption(f"Averaged across {overall.get('curve_count', 0)} curve(s)")
        
        o1, o2, o3 = st.columns(3)
        with o1:
            st.metric("Delta Value", f"{overall['delta_value']:.4f}",
                      help="Mean absolute error averaged across all curves (in axis units)")
        with o2:
            st.metric("Delta Norm", f"{overall['delta_norm']:.4%}",
                      help="Normalized error averaged across all curves (lower is better)")
        with o3:
            st.metric("Delta P95", f"{overall['delta_p95']:.4f}",
                      help="Worst 95th-percentile error across all curves")
        
        o4, o5, o6 = st.columns(3)
        with o4:
            st.metric("IoU", f"{overall['iou']:.4f}",
                      help="Average Intersection over Union across all curves (1.0 = perfect)")
        with o5:
            st.metric("Precision", f"{overall['precision']:.4f}",
                      help="Average precision across all curves")
        with o6:
            st.metric("Recall", f"{overall['recall']:.4f}",
                      help="Average recall across all curves")
    
    st.subheader("🎨 Detected Curves")
    
    curves = results.get('curves', {})
    if curves:
        for color, curve_data in curves.items():
            with st.expander(f"🔹 {color.upper()} - {curve_data.get('label', 'Unknown')}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Original Points",
                        curve_data.get('original_point_count', 0)
                    )
                
                with col2:
                    st.metric(
                        "Cleaned Points",
                        curve_data.get('cleaned_point_count', 0)
                    )
                
                with col3:
                    fit_result = curve_data.get('fit_result', {})
                    r_squared = fit_result.get('r_squared', 0)
                    st.metric(
                        "R² Score",
                        f"{r_squared:.4f}"
                    )
                
                if 'fit_result' in curve_data and curve_data['fit_result']:
                    fit = curve_data['fit_result']
                    
                    if fit.get('coefficients'):
                        st.write("**Polynomial Coefficients:**")
                        st.code(f"y = {fit.get('equation', 'N/A')}")
                        
                        coeffs = fit.get('coefficients', [])
                        if isinstance(coeffs, list):
                            for i, coeff in enumerate(coeffs):
                                st.write(f"  c{i} = {coeff:.6f}")
                    elif fit.get('equation'):
                        st.write(f"**Fit Type:** {fit.get('equation')}")
                    
                    if fit.get('error'):
                        st.error(f"**Error:** {fit['error']}")
                
                # ── Quality Metrics ──
                metrics = curve_data.get('metrics', {})
                if metrics and metrics.get('delta_value') is not None:
                    st.write("---")
                    st.write("**📐 Quality Metrics**")
                    
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Delta Value", f"{metrics['delta_value']:.4f}",
                                  help="Mean absolute error between extracted data points and fitted curve (in axis units)")
                    with m2:
                        st.metric("Delta Norm", f"{metrics['delta_norm']:.4%}",
                                  help="Delta normalized by Y-axis range — scale-independent error (lower is better)")
                    with m3:
                        st.metric("Delta P95", f"{metrics['delta_p95']:.4f}",
                                  help="95th percentile of absolute errors — worst-case error excluding outliers")
                    
                    m4, m5, m6 = st.columns(3)
                    with m4:
                        st.metric("IoU", f"{metrics['iou']:.4f}",
                                  help="Intersection over Union — overlap between fitted curve and actual data bands (1.0 = perfect)")
                    with m5:
                        st.metric("Precision", f"{metrics['precision']:.4f}",
                                  help="Fraction of fitted curve regions that have actual data nearby (1.0 = no hallucinated curve segments)")
                    with m6:
                        st.metric("Recall", f"{metrics['recall']:.4f}",
                                  help="Fraction of actual data regions captured by the fitted curve (1.0 = no data missed)")
    else:
        st.warning("No curves detected in the image.")
    
    # ── Toggleable overlay comparison at the bottom ──
    output_graphs = results.get('output_graphs', {})
    overlay_img = output_graphs.get('overlay', '')
    if input_img and Path(input_img).exists() and overlay_img and Path(overlay_img).exists():
        st.subheader("🔍 Overlay Comparison (click to toggle)")

        # Session-state key to track which image is shown
        toggle_key = "overlay_toggle"
        if toggle_key not in st.session_state:
            st.session_state[toggle_key] = "output"  # start with overlay

        if st.button(
            "Show Input Image" if st.session_state[toggle_key] == "output" else "Show Output Overlay",
            key="overlay_toggle_btn",
        ):
            st.session_state[toggle_key] = (
                "input" if st.session_state[toggle_key] == "output" else "output"
            )

        if st.session_state[toggle_key] == "output":
            st.image(overlay_img, caption="Digitized Output (Overlay)",
                     width=600)
        else:
            st.image(input_img, caption="Original Input",
                     width=600)

    # Download results
    if results.get('output_file'):
        with open(results['output_file'], 'r') as f:
            json_data = json.dumps(json.load(f), indent=2)
        
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="📥 Download Results (JSON)",
                data=json_data,
                file_name=f"curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        with col_dl2:
            # CSV export of curve data
            csv_data = _build_csv_export(results)
            if csv_data:
                st.download_button(
                    label="📥 Download Curves (CSV)",
                    data=csv_data,
                    file_name=f"curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

    # ── Per-Curve Coordinate Export ──
    _display_coordinate_export(results)

    # ── Pipeline Info ──
    detected_mode = results.get('detected_mode', '')
    extraction_method = results.get('extraction_method', '')
    if detected_mode or extraction_method:
        st.subheader("🔧 Pipeline Info")
        info_cols = st.columns(3)
        with info_cols[0]:
            st.write(f"**Mode:** {detected_mode}")
        with info_cols[1]:
            st.write(f"**Extraction:** {extraction_method or 'color'}")
        with info_cols[2]:
            st.write(f"**Grayscale:** {results.get('grayscale_mode', False)}")

    # ── Validation warnings ──
    validation = results.get('validation', {})
    if validation:
        warning_msg = validation.get('warning', '')
        if warning_msg:
            st.warning(f"⚠️ {warning_msg}")
        with st.expander("📊 Mapping Validation", expanded=bool(warning_msg)):
            v1, v2 = st.columns(2)
            with v1:
                st.write(f"**Data Y range:** "
                         f"{validation.get('y_data_min', '?')} – "
                         f"{validation.get('y_data_max', '?')}")
            with v2:
                st.write(f"**Axis Y range:** "
                         f"{validation.get('y_axis_min', '?')} – "
                         f"{validation.get('y_axis_max', '?')}")
            st.write(f"**Y coverage:** {validation.get('y_coverage_pct', '?')}%")
            pa_px = validation.get('plot_area_pixels', [])
            if pa_px:
                st.write(f"**Plot area (px):** L={pa_px[0]}, T={pa_px[1]}, "
                         f"R={pa_px[2]}, B={pa_px[3]}")

    # ── Debug Overlay (if enabled) ──
    show_debug = False
    if hasattr(st, 'session_state'):
        settings = st.session_state.get("pipeline_settings", {})
        show_debug = settings.get("show_debug", False)
    
    if show_debug:
        _display_debug_overlay(results)


def _display_coordinate_export(results: Dict[str, Any]) -> None:
    """Per-curve coordinate lists with preview + per-curve CSV download."""
    curves = results.get('curves', {})
    if not curves:
        return

    st.subheader("📋 Coordinate Export (per curve)")

    for color, curve_data in curves.items():
        if not isinstance(curve_data, dict):
            continue

        label = curve_data.get('label', color)

        # Gather available coordinate sources
        fit = curve_data.get('fit_result', {}) or {}
        fitted_pts = fit.get('fitted_points', [])
        axis_coords = curve_data.get('axis_coords', [])
        pixel_coords = curve_data.get('pixel_coords', [])

        # Pick best source
        if fitted_pts:
            pts = [(pt['x'], pt['y']) for pt in fitted_pts]
            source = 'fitted'
        elif axis_coords:
            pts = [(pt[0], pt[1]) for pt in axis_coords]
            source = 'axis'
        elif pixel_coords:
            pts = [(pt[0], pt[1]) for pt in pixel_coords]
            source = 'pixel'
        else:
            continue

        with st.expander(f"🔹 {color.upper()} — {label}  ({len(pts)} pts, {source})", expanded=False):
            # Preview first 10
            preview = pts[:10]
            header = "x,y" if source != 'pixel' else "px_x,px_y"
            rows = [header] + [f"{x:.4f},{y:.4f}" for x, y in preview]
            if len(pts) > 10:
                rows.append(f"... ({len(pts) - 10} more)")
            st.code("\n".join(rows), language="csv")

            # Also show pixel coords if we have data coords
            if source in ('fitted', 'axis') and pixel_coords:
                st.caption(f"Raw pixel coordinates: {len(pixel_coords)} points")

            # Per-curve CSV download
            csv_buf = io.StringIO()
            writer = csv.writer(csv_buf)
            writer.writerow(['curve', 'label', 'source', 'x', 'y'])
            for x, y in pts:
                writer.writerow([color, label, source, round(x, 6), round(y, 6)])
            # Append pixel coords as separate rows if available
            if source != 'pixel' and pixel_coords:
                for pt in pixel_coords:
                    px, py = (pt[0], pt[1]) if isinstance(pt, (list, tuple)) else (pt.get('x', 0), pt.get('y', 0))
                    writer.writerow([color, label, 'pixel', int(px), int(py)])

            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.download_button(
                label=f"📥 CSV — {label}",
                data=csv_buf.getvalue(),
                file_name=f"{color}_{ts}.csv",
                mime="text/csv",
                key=f"dl_csv_{color}_{ts}",
            )

            # Per-curve JSON download
            json_payload = {
                "curve": color,
                "label": label,
                "source": source,
                "point_count": len(pts),
                "data_points": [{"x": round(x, 6), "y": round(y, 6)} for x, y in pts],
            }
            if source != 'pixel' and pixel_coords:
                json_payload["pixel_points"] = [
                    {"x": int(pt[0] if isinstance(pt, (list, tuple)) else pt.get('x', 0)),
                     "y": int(pt[1] if isinstance(pt, (list, tuple)) else pt.get('y', 0))}
                    for pt in pixel_coords
                ]
            st.download_button(
                label=f"📥 JSON — {label}",
                data=json.dumps(json_payload, indent=2),
                file_name=f"{color}_{ts}.json",
                mime="application/json",
                key=f"dl_json_{color}_{ts}",
            )


def _build_csv_export(results: Dict[str, Any]) -> Optional[str]:
    """Build CSV string from extracted curve data."""
    curves = results.get('curves', {})
    if not curves:
        return None
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow(['curve', 'label', 'source', 'x', 'y'])
    
    for color, curve_data in curves.items():
        if not isinstance(curve_data, dict):
            continue
        label = curve_data.get('label', color)
        
        # Prefer fitted_points (data coords)
        fit = curve_data.get('fit_result', {})
        fitted_pts = fit.get('fitted_points', []) if isinstance(fit, dict) else []
        if fitted_pts:
            for pt in fitted_pts:
                writer.writerow([color, label, 'fitted', 
                               round(pt['x'], 4), round(pt['y'], 4)])
        else:
            # Fallback to axis_coords or pixel_coords
            axis_coords = curve_data.get('axis_coords', [])
            if axis_coords:
                for pt in axis_coords:
                    writer.writerow([color, label, 'axis',
                                   round(pt[0], 4), round(pt[1], 4)])
            else:
                pixel_coords = curve_data.get('pixel_coords', [])
                for pt in pixel_coords:
                    writer.writerow([color, label, 'pixel',
                                   int(pt[0]), int(pt[1])])
    
    return output.getvalue()


def _display_debug_overlay(results: Dict[str, Any]) -> None:
    """Display debug information and overlays."""
    st.subheader("🔍 Debug Overlay")
    
    # Plot area bounds
    pa = results.get('plot_area', {})
    if pa:
        st.write(f"**Plot Area:** left={pa.get('left')}, top={pa.get('top')}, "
                f"right={pa.get('right')}, bottom={pa.get('bottom')}")
    
    # Per-curve debug info
    curves = results.get('curves', {})
    for color, curve_data in curves.items():
        if not isinstance(curve_data, dict):
            continue
        with st.expander(f"Debug: {color}", expanded=False):
            st.write(f"**Extraction mode:** {curve_data.get('extraction_mode', 'color')}")
            st.write(f"**Raw pixels:** {curve_data.get('original_point_count', 0)}")
            st.write(f"**Cleaned:** {curve_data.get('cleaned_point_count', 0)}")
            
            # Plot area used for this curve
            cpa = curve_data.get('plot_area', [])
            if cpa:
                st.write(f"**Curve plot_area:** {cpa}")
            
            # Fit info
            fit = curve_data.get('fit_result', {})
            if fit and isinstance(fit, dict):
                st.write(f"**Fit degree:** {fit.get('degree', 'N/A')}")
                st.write(f"**R²:** {fit.get('r_squared', 'N/A')}")
                st.write(f"**Fitted points:** {len(fit.get('fitted_points', []))}")
    
    # Debug images from env var
    import os
    debug_dir = os.environ.get("CURVE_DEBUG_IMAGES", "")
    if debug_dir and Path(debug_dir).exists():
        st.write("**Debug images:**")
        debug_files = sorted(Path(debug_dir).glob("*.png"))
        for df in debug_files[:10]:
            st.image(str(df), caption=df.name, width=400)


def display_processing_summary(axis_info: Dict, features: Dict) -> str:
    """
    Generate a summary message for processed image.
    
    Args:
        axis_info: Axis information from Gemini
        features: Detected features from Gemini
        
    Returns:
        Summary text
    """
    curve_count = len(features.get('curves', []))
    
    summary = f"""✅ Image analysis complete!

**Axis Information:**
- X-axis: {axis_info.get('xMin')} to {axis_info.get('xMax')} ({axis_info.get('xUnit', 'units')})
- Y-axis: {axis_info.get('yMin')} to {axis_info.get('yMax')} ({axis_info.get('yUnit', 'units')})

**Curves Detected:** {curve_count}"""
    
    return summary
