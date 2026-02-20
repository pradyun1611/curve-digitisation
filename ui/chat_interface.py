"""
Chat interface module for Streamlit UI.

Handles chat message display and conversation history.
"""

import streamlit as st


def display_chat_message(role: str, content: str) -> None:
    """
    Display a single chat message.
    
    Args:
        role: Role of the speaker ('user' or 'bot')
        content: Message content
    """
    if role == "user":
        st.markdown(f"""
            <div class="stMessage user-message">
                <strong>You:</strong> {content}
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="stMessage bot-message">
                <strong>Bot:</strong> {content}
            </div>
        """, unsafe_allow_html=True)


def display_chat_history(chat_history: list) -> None:
    """
    Display entire chat conversation history.
    
    Args:
        chat_history: List of message dictionaries with 'role' and 'content'
    """
    for message in chat_history:
        display_chat_message(message['role'], message['content'])


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'openai_client' not in st.session_state:
        st.session_state.openai_client = None
    
    if 'api_key_set' not in st.session_state:
        st.session_state.api_key_set = False
    
    if 'results_history' not in st.session_state:
        st.session_state.results_history = []
