"""
Sidebar configuration module for Streamlit UI.

Handles sidebar setup and API key configuration.
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
        - Export results as JSON
        
        **How to use:**
        1. Configure Azure OpenAI credentials via environment variables
        2. Type a message or upload an image
        3. For image processing, mention the image in your query
        """)
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.session_state.results_history = []
            st.rerun()
    
    # Load credentials from environment variables
    api_key = os.getenv('OPENAI_API_KEY', '')
    endpoint = os.getenv('AZURE_ENDPOINT', '')
    deployment_name = os.getenv('AZURE_DEPLOYMENT_NAME', '')
    output_dir = "./output/"
    
    return api_key, endpoint, deployment_name, output_dir
