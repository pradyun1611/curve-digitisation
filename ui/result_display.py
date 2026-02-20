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
                    
                    if fit.get('error'):
                        st.error(f"**Error:** {fit['error']}")
    else:
        st.warning("No curves detected in the image.")
    
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
