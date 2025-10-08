from datetime import datetime
from typing import Generator, List, Tuple

import streamlit as st

from modules.chat.types import Interaction
from modules.config.pricing import PricingCalculator
from modules.config.settings import Config


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
            
            # Model selection - dropdown with available models
            available_models = PricingCalculator.get_available_models(provider)
            default_model = Config.get_default_model(provider)
            
            if available_models:
                # If we have models in the pricing list, use selectbox
                if default_model in available_models:
                    default_index = available_models.index(default_model)
                else:
                    default_index = 0
                
                model = st.selectbox(
                    "Model",
                    options=available_models,
                    index=default_index,
                    help="Select the model to use"
                )
            else:
                # Fallback to text input if no models are available
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
    def display_raw_interactions(interactions: List[Interaction]):
        """Display raw request data sent to LLM"""
        if not interactions:
            st.info("💬 No interactions yet. Send a message to see raw request data.")
            return
        
        RawMessageViewer._display_requests(interactions)
    
    @staticmethod
    def _display_requests(interactions: List[Interaction]):
        """Display request details"""
        st.subheader("📤 Raw details")
        
        if not interactions:
            st.write("No requests to display")
            return
        
        # Select interaction
        interaction_options = [
            f"#{idx + 1} - {datetime.fromtimestamp(i['timestamp']).strftime('%H:%M:%S')} - {i['request']['model']}"
            for idx, i in enumerate(interactions)
        ]
        
        selected_idx = st.selectbox(
            "Select interaction:",
            list(range(len(interactions))),
            format_func=lambda x: str(interaction_options[x]),
            index=len(interactions) - 1 if interactions else 0
        )
        
        # Early return if out of bounds (defensive programming)
        if selected_idx >= len(interactions):
            return
            
        interaction = interactions[selected_idx]
        request = interaction["request"]
        
        # Display request metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Provider:** {request['provider']}")
        with col2:
            st.write(f"**Model:** {request['model']}")
        with col3:
            st.write(f"**Stream:** {request['stream']}")
        
        # Display token usage if available
        RawMessageViewer._display_token_usage(interaction)
        
        # Display messages
        st.write("### Messages Sent")
        messages = request["messages"]
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
    def _display_token_usage(interaction: Interaction):
        """Display token usage information if available"""
        if interaction["type"] == "complete":
            response = interaction["response"]
            raw = response.get("raw")
            
            if raw and "usage" in raw:
                usage = raw["usage"]
                if usage:
                    col1, col2, col3 = st.columns(3)
                    
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")
                    total_tokens = usage.get("total_tokens")
                    
                    with col1:
                        if prompt_tokens is not None:
                            st.metric("Input Tokens", f"{prompt_tokens:,}")
                        else:
                            st.metric("Input Tokens", "N/A")
                    
                    with col2:
                        if completion_tokens is not None:
                            st.metric("Output Tokens", f"{completion_tokens:,}")
                        else:
                            st.metric("Output Tokens", "N/A")
                    
                    with col3:
                        if total_tokens is not None:
                            st.metric("Total Tokens", f"{total_tokens:,}")
                        else:
                            st.metric("Total Tokens", "N/A")
                    
                    # Display cost if pricing information is available
                    if prompt_tokens is not None and completion_tokens is not None:
                        request = interaction["request"]
                        provider = request["provider"]
                        model = request["model"]
                        
                        pricing_info = PricingCalculator.get_model_pricing(provider, model)
                        
                        if pricing_info is not None:
                            cost = PricingCalculator.calculate_cost(
                                provider=provider,
                                model=model,
                                input_tokens=prompt_tokens,
                                output_tokens=completion_tokens
                            )
                            
                            if cost is not None:
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    input_cost = (prompt_tokens / 1_000_000) * pricing_info["input_price"]
                                    st.metric(
                                        "Input Cost",
                                        PricingCalculator.format_cost(input_cost),
                                        help=f"${pricing_info['input_price']}/1M tokens"
                                    )
                                
                                with col2:
                                    output_cost = (completion_tokens / 1_000_000) * pricing_info["output_price"]
                                    st.metric(
                                        "Output Cost",
                                        PricingCalculator.format_cost(output_cost),
                                        help=f"${pricing_info['output_price']}/1M tokens"
                                    )
                                
                                with col3:
                                    st.metric(
                                        "Total Cost",
                                        PricingCalculator.format_cost(cost),
                                        help="Sum of input and output costs"
                                    )
                        else:
                            st.info(f"ℹ️ Pricing information not available for model: {model}")
        elif interaction["type"] == "streaming":
            # For streaming, we don't have usage info typically
            st.info("ℹ️ Token usage information is not available for streaming responses")
    
    @staticmethod
    def display_streaming_status(chunk_count: int):
        """Display real-time streaming status"""
        return st.empty().info(f"🔄 Streaming... ({chunk_count} chunks received)")
