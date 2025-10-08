from typing import Generator, Tuple

import streamlit as st

from config import Config


class Sidebar:
    """Encapsulates sidebar UI components"""
    
    @staticmethod
    def render() -> Tuple[str, str, str, bool, bool]:
        """Render sidebar and return settings"""
        with st.sidebar:
            st.header("Settings")
            
            # Provider selection
            provider = st.selectbox(
                "Provider", 
                Config.get_provider_names(),
                help="Select the LLM provider to use"
            )
            
            # Model selection
            default_model = Config.get_default_model(provider)
            model = st.text_input(
                "Model", 
                value=default_model,
                help="Enter the model name (e.g., gpt-4, claude-3)"
            )
            
            # System prompt
            system_prompt = st.text_area(
                "System prompt",
                value=Config.DEFAULT_SYSTEM_PROMPT,
                height=80,
                help="Instructions that guide the assistant's behavior"
            )
            
            # Stream toggle
            stream = st.toggle(
                "Stream responses", 
                value=True,
                help="Enable real-time streaming of responses"
            )
            
            # Clear button
            clear_pressed = st.button(
                "Clear chat",
                type="secondary",
                use_container_width=True
            )
            
            return provider, model, system_prompt, stream, clear_pressed


class ChatUI:
    """Handles chat display and interaction"""
    
    @staticmethod
    def display_message(role: str, content: str):
        """Display a single chat message"""
        with st.chat_message(role):
            st.markdown(content)
    
    @staticmethod
    def display_streaming_response(response_generator: Generator[str, None, None]) -> str:
        """Display a streaming response with incremental updates"""
        placeholder = st.empty()
        accumulated = ""
        
        try:
            for chunk in response_generator:
                accumulated += chunk
                placeholder.markdown(accumulated)
        except Exception as e:
            # If streaming fails, display what we have so far
            if accumulated:
                placeholder.markdown(accumulated)
            raise e
        
        return accumulated
    
    @staticmethod
    def display_response(content: str) -> str:
        """Display a non-streaming response"""
        st.markdown(content)
        return content
    
    @staticmethod
    def display_error(error: Exception, show_details: bool = True):
        """Display error message with optional details"""
        error_msg = f"⚠️ Error: {str(error)}"
        st.error(error_msg)
        
        if show_details:
            import traceback
            with st.expander("Error Details", expanded=False):
                st.code(traceback.format_exc(), language="python")