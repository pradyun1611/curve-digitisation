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
        Tuple of (api_key, output_dir)
    """
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Azure OpenAI Configuration
        st.subheader("🔐 Azure OpenAI Configuration")
        
        api_key = st.text_input(
            "Azure OpenAI API Key",
            type="password",
            value=os.getenv('OPENAI_API_KEY', ''),
            help="Your Azure OpenAI API key"
        )
        
        endpoint = st.text_input(
            "Azure Endpoint",
            value=os.getenv('AZURE_ENDPOINT', ''),
            help="e.g., https://your-resource.openai.azure.com/"
        )
        
        deployment_name = st.text_input(
            "Deployment Name",
            value=os.getenv('AZURE_DEPLOYMENT_NAME', ''),
            help="e.g., gpt-5-chat, gpt-4, etc."
        )
        
        # Output directory
        output_dir = st.text_input(
            "Output Directory",
            value="./output/",
            help="Where to save processing results"
        )
        
        st.divider()
        
        # Info section
        st.subheader("📚 About")
        st.write("""
        This chatbot helps you extract and digitize curves from performance chart images.
        
        **Features:**
        - Extract axis information and units
        - Identify and isolate curves by color
        - Fit polynomial curves to extracted data
        - Clean noisy data with RANSAC filtering
        - Export results as JSON
        
        **How to use:**
        1. Enter your OpenAI API key
        2. Type a message or upload an image
        3. For image processing, mention the image in your query
        """)
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.session_state.results_history = []
            st.rerun()
    
    return api_key, endpoint, deployment_name, output_dir
