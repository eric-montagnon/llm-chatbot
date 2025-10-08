from datetime import datetime
from typing import Any, Dict, Generator, List, Tuple

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


class RawMessageViewer:
    """Component for displaying raw LLM interactions"""
    
    @staticmethod
    def display_raw_interactions(interactions: List[Dict[str, Any]]):
        """Display raw request data sent to LLM"""
        if not interactions:
            st.info("💬 No interactions yet. Send a message to see raw request data.")
            return
        
        RawMessageViewer._display_requests(interactions)
    
    @staticmethod
    def _display_requests(interactions: List[Dict[str, Any]]):
        """Display request details"""
        st.subheader("📤 Request Messages to LLM")
        
        if not interactions:
            st.write("No requests to display")
            return
        
        # Select interaction
        interaction_options = [
            f"#{idx + 1} - {datetime.fromtimestamp(i.get('timestamp', 0)).strftime('%H:%M:%S')} - {i.get('request', {}).get('model', 'N/A')}"
            for idx, i in enumerate(interactions)
        ]
        
        selected_idx = st.selectbox(
            "Select interaction:",
            range(len(interactions)),
            format_func=lambda x: interaction_options[x],
            index=len(interactions) - 1 if interactions else 0
        )
        
        if selected_idx is not None and selected_idx < len(interactions):
            interaction = interactions[selected_idx]
            request = interaction.get("request", {})
            
            # Display request metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Provider:** {request.get('provider', 'N/A')}")
            with col2:
                st.write(f"**Model:** {request.get('model', 'N/A')}")
            with col3:
                st.write(f"**Stream:** {request.get('stream', False)}")
            
            # Display messages
            st.write("### Messages Sent")
            messages = request.get("messages", [])
            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                
                if role == "system":
                    with st.expander(f"🔧 System", expanded=False):
                        st.text(content)
                elif role == "user":
                    with st.expander(f"👤 User", expanded=True):
                        st.text(content)
                elif role == "assistant":
                    with st.expander(f"🤖 Assistant", expanded=True):
                        st.text(content)
            
            # Raw JSON view
            with st.expander("📄 Raw Request JSON", expanded=False):
                st.json(request)
    
    @staticmethod
    def display_streaming_status(chunk_count: int):
        """Display real-time streaming status"""
        return st.empty().info(f"🔄 Streaming... ({chunk_count} chunks received)")
