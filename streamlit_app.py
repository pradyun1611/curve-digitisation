"""
Streamlit UI for Curve Digitization Chatbot

Interactive web interface for the curve digitization chatbot using Streamlit.

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import json
import os
import base64
from pathlib import Path
from datetime import datetime
from typing import Optional
import io
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from core.openai_client import OpenAIClient
from core.image_processor import CurveDigitizer
from ui.sidebar import setup_sidebar
from ui.chat_interface import display_chat_message, display_chat_history, initialize_session_state
from ui.result_display import display_image_results, display_processing_summary
from ui.click_canvas import render_anchor_canvas




# Page configuration
st.set_page_config(
    page_title="Curve Digitization Chatbot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Adaptive dark/light mode
st.markdown("""
    <style>
    .stMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    /* Light mode */
    @media (prefers-color-scheme: light) {
        .user-message {
            background-color: #bbdefb;
            border-left: 4px solid #1976d2;
            color: #0d47a1;
        }
        .bot-message {
            background-color: #c8e6c9;
            border-left: 4px solid #388e3c;
            color: #1b5e20;
        }
    }
    
    /* Dark mode */
    @media (prefers-color-scheme: dark) {
        .user-message {
            background-color: #1565c0;
            border-left: 4px solid #64b5f6;
            color: #e3f2fd;
        }
        .bot-message {
            background-color: #2e7d32;
            border-left: 4px solid #81c784;
            color: #e8f5e9;
        }
    }
    
    .success {
        color: #4CAF50;
        font-weight: bold;
    }
    .error {
        color: #f44336;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


def get_openai_client(api_key: str, endpoint: str, deployment_name: str) -> Optional[OpenAIClient]:
    """Get or create OpenAI client."""
    if st.session_state.openai_client is None and api_key and endpoint and deployment_name:
        try:
            st.session_state.openai_client = OpenAIClient(api_key, endpoint, deployment_name)
            st.session_state.api_key_set = True
            return st.session_state.openai_client
        except Exception as e:
            st.error(f"Failed to initialize Azure OpenAI client: {e}")
            return None
    return st.session_state.openai_client


def save_image_as_base64(image_file) -> str:
    """Convert uploaded image to base64."""
    image_bytes = image_file.read()
    return base64.b64encode(image_bytes).decode('utf-8')


def process_image(client: OpenAIClient, image_file, user_query: str, output_dir: str) -> dict:
    """Process uploaded image and extract curves."""
    try:
        # Save uploaded image temporarily (cross-platform)
        import tempfile
        temp_dir = Path(tempfile.gettempdir())
        temp_path = str(temp_dir / image_file.name)
        
        with open(temp_path, "wb") as f:
            f.write(image_file.getbuffer())
        
        # Encode to base64
        image_base64 = save_image_as_base64(io.BytesIO(image_file.getbuffer()))
        
        # Extract axis information
        with st.spinner("Extracting axis information..."):
            axis_info = client.extract_axis_info(image_base64, user_query)
        
        # Apply calibration overrides from sidebar
        settings = st.session_state.get("pipeline_settings", {})
        cal_overrides = settings.get("calibration_overrides", {})
        if cal_overrides:
            for k, v in cal_overrides.items():
                if v is not None:
                    axis_info[k] = v
        
        # Extract curve features
        with st.spinner("Detecting curves..."):
            features = client.extract_curve_features(image_base64)
        
        # Digitize curves and generate graphs
        with st.spinner("Fitting polynomial curves and generating graphs..."):
            digitizer = CurveDigitizer(axis_info)
            
            # Get pipeline settings from sidebar
            mode = settings.get("mode", "auto")
            ignore_dashed = settings.get("ignore_dashed", True)
            smoothing_strength = settings.get("smoothing_strength", 0)
            use_skeleton = settings.get("use_skeleton", True)
            anchors = settings.get("anchors", None)
            target_curves = settings.get("target_curves", 0)
            dashed_threshold = settings.get("dashed_threshold", 0.45)
            text_threshold = settings.get("text_threshold", 0.50)
            plot_area_override = settings.get("plot_area_override", None)
            
            results = digitizer.process_curve_image(
                temp_path, features, output_dir,
                mode=mode,
                anchors=anchors,
                ignore_dashed=ignore_dashed,
                smoothing_strength=smoothing_strength,
                use_skeleton_bw=use_skeleton,
                target_curves=target_curves,
                dashed_threshold=dashed_threshold,
                text_threshold=text_threshold,
                plot_area_override=plot_area_override,
            )
        
        # Attach pipeline settings metadata for export provenance
        results['pipeline_settings'] = {
            'mode': mode,
            'ignore_dashed': ignore_dashed,
            'smoothing_strength': smoothing_strength,
            'use_skeleton': use_skeleton,
            'calibration_overrides': cal_overrides or {},
            'version': '2.0',
        }

        # Save JSON results in the per-instance output folder
        instance_dir = Path(results.get('instance_dir', output_dir))
        output_file = instance_dir / "curve_digitization.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        results['output_file'] = str(output_file)
        
        # Copy input image to instance folder so it survives temp cleanup
        import shutil
        input_copy = str(instance_dir / f"input_{Path(temp_path).name}")
        shutil.copy2(temp_path, input_copy)
        results['input_image_path'] = input_copy
        
        # Clean up temp file
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        
        return results
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return {'_error': str(e)}


def main():
    """Main Streamlit app."""
    initialize_session_state()
    
    # Sidebar configuration
    api_key, endpoint, deployment_name, output_dir = setup_sidebar()
    
    # Initialize client if API key and Azure config provided
    if api_key and endpoint and deployment_name:
        client = get_openai_client(api_key, endpoint, deployment_name)
    else:
        client = None
    
    # Main content
    st.title("📈 Curve Digitization Chatbot")
    
    if not client:
        st.warning("⚠️ Please enter your Azure OpenAI credentials in the sidebar to get started.")
        return
    
    # Chat messages display
    st.subheader("💬 Conversation")
    
    # Display chat history
    display_chat_history(st.session_state.chat_history)
    
    # Input section
    st.divider()
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Your message:",
            placeholder="Type your query or describe what you want to do with an image...",
            key="user_input"
        )
    
    with col2:
        send_button = st.button("Send", key="send_button", use_container_width=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📸 Upload an image (PNG, JPG, etc.)",
        type=['png', 'jpg', 'jpeg', 'bmp', 'gif'],
        key="image_uploader"
    )

    # ── Reset anchors when a different image is uploaded ──
    if uploaded_file is not None:
        _current_name = uploaded_file.name
        if st.session_state.get("_last_uploaded_file") != _current_name:
            st.session_state["_last_uploaded_file"] = _current_name
            st.session_state.pop("click_curves", None)
            st.session_state.pop("picking", None)
            st.session_state.pop("last_click_xy", None)

    # ── Click-to-place anchor canvas ──
    if uploaded_file is not None:
        with st.expander("📍 Click-to-Place Anchors", expanded=False):
            st.caption(
                "Add curves below, then click 📌 Start / 📌 End and click "
                "on the image to place anchor points.  Anchors are sent to "
                "the pipeline automatically when you press **Send**."
            )
            anchor_pairs = render_anchor_canvas(uploaded_file.getvalue())
            uploaded_file.seek(0)  # reset after reading
    
    # Process user input
    if send_button and user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # ── Decision logic ──
        # When a file IS uploaded, always process it as an image—regardless
        # of how the user phrased their prompt.  Intent classification is
        # only used when NO file is present.
        if uploaded_file:
            st.session_state.chat_history.append({
                'role': 'bot',
                'content': '🔄 Processing image...'
            })
            
            # Process image
            results = process_image(client, uploaded_file, user_input, output_dir)
            
            if results and '_error' not in results:
                # Update chat with summary
                axis_info = results.get('axis_info', {})
                features_summary = {
                    'curves': [{'color': c} for c in results.get('curves', {}).keys()]
                }
                summary = display_processing_summary(axis_info, features_summary)
                
                st.session_state.chat_history.append({
                    'role': 'bot',
                    'content': summary
                })
                
                st.session_state.results_history.append(results)
            else:
                err_msg = results.get('_error', 'Unknown error') if results else 'Unknown error'
                st.session_state.chat_history.append({
                    'role': 'bot',
                    'content': f'❌ Failed to process image: {err_msg}'
                })
        else:
            # No file uploaded – check intent to give helpful guidance
            intent = client.classify_intent(user_input)
            if intent == 'imageprocessing':
                response = ("It looks like you want to process an image. "
                            "Please upload an image file using the uploader above and try again.")
                st.session_state.chat_history.append({
                    'role': 'bot',
                    'content': response
                })
            else:
                response = client.get_general_response(user_input)
                st.session_state.chat_history.append({
                    'role': 'bot',
                    'content': response
                })
        
        st.rerun()
    
    # Display latest image results if available
    if st.session_state.results_history:
        st.divider()
        st.subheader("📊 Latest Results")
        display_image_results(st.session_state.results_history[-1])


if __name__ == '__main__':
    main()
