"""
Result display module for Streamlit UI.

Handles visualization of image processing results.
"""

import streamlit as st
import json
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
    
    # ── Display the generated digitized output graph ──
    output_graphs = results.get('output_graphs', {})
    if output_graphs:
        st.subheader("📊 Reconstructed Digitized Curves")
        for graph_name, graph_path in output_graphs.items():
            import os
            if os.path.exists(graph_path):
                st.image(graph_path, caption=graph_name.replace('_', ' ').title(),
                         use_container_width=True)

    # Download results
    if results.get('output_file'):
        with open(results['output_file'], 'r') as f:
            json_data = json.dumps(json.load(f), indent=2)
        
        st.download_button(
            label="📥 Download Results (JSON)",
            data=json_data,
            file_name=f"curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


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
