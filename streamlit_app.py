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
        # Save uploaded image temporarily
        temp_path = f"/tmp/{image_file.name}"
        Path("/tmp").mkdir(exist_ok=True)
        
        with open(temp_path, "wb") as f:
            f.write(image_file.getbuffer())
        
        # Encode to base64
        image_base64 = save_image_as_base64(io.BytesIO(image_file.getbuffer()))
        
        # Extract axis information
        with st.spinner("Extracting axis information..."):
            axis_info = client.extract_axis_info(image_base64, user_query)
        
        # Extract curve features
        with st.spinner("Detecting curves..."):
            features = client.extract_curve_features(image_base64)
        
        # Digitize curves
        with st.spinner("Fitting polynomial curves..."):
            digitizer = CurveDigitizer(axis_info)
            results = digitizer.process_curve_image(temp_path, features)
        
        # Save results
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(output_dir) / f"curve_digitization_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        results['output_file'] = str(output_file)
        
        # Clean up temp file
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        
        return results
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None


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
    
    # Process user input
    if send_button and user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # Classify intent
        intent = client.classify_intent(user_input)
        
        # Process based on intent
        if intent == 'imageprocessing':
            if uploaded_file:
                st.session_state.chat_history.append({
                    'role': 'bot',
                    'content': '🔄 Processing image...'
                })
                
                # Process image
                results = process_image(client, uploaded_file, user_input, output_dir)
                
                if results:
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
                    st.session_state.chat_history.append({
                        'role': 'bot',
                        'content': '❌ Failed to process image. Please check the image format and try again.'
                    })
            else:
                response = "I detected this is about image processing, but I don't see an image. Please upload an image file to process."
                st.session_state.chat_history.append({
                    'role': 'bot',
                    'content': response
                })
        else:  # General response
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
