from typing import Generator, Tuple

import streamlit as st
from langchain.messages import AIMessage

from modules.config.pricing import PricingCalculator
from modules.config.settings import Config


class Sidebar:
    """Encapsulates sidebar UI components"""
    
    @staticmethod
    def render() -> Tuple[str, str, str, bool]:
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
            
            
            # Clear button
            clear_pressed = st.button(
                "Clear chat",
                type="secondary",
                use_container_width=True
            )
            
            return provider, model, system_prompt, clear_pressed


class ChatUI:
    """Handles chat display and interaction"""
    
    @staticmethod
    def display_message(role: str, content: str):
        """Display a single chat message"""
        with st.chat_message(role):
            st.markdown(content)

    @staticmethod
    def display_ai_message_with_costs(message: AIMessage):
        """Display an AI message with token usage and cost information
        
        Args:
            message: The AIMessage object containing content, usage metadata, and response metadata
        """
        with st.chat_message("assistant"):
            # Create columns: main content (wider) and cost info (narrower)
            col1, col2 = st.columns([4, 1])
            
            with col1:
                # Display the message content
                if message.content:
                    st.markdown(message.content)
            
            with col2:
                # Display token usage and cost if available
                usage = message.usage_metadata
                if usage:
                    input_tokens = usage.get('input_tokens', 0)
                    output_tokens = usage.get('output_tokens', 0)
                    
                    # Extract model from response_metadata
                    response_metadata = message.response_metadata
                    model_name = response_metadata.get('model_name', '')
                    
                    # Calculate cost using just the model name
                    cost = PricingCalculator.calculate_cost(
                        model=model_name,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens
                    ) if model_name else None

                    if cost is not None:
                        formatted_cost = PricingCalculator.format_cost(cost)
                        st.markdown(f"**💰 {formatted_cost}**")
                    st.caption(f"🔼 {input_tokens:,} in")
                    st.caption(f"🔽 {output_tokens:,} out")
    
    @staticmethod
    def display_tool_calls(message: AIMessage, response: str = ""):
        """Display tool call executions with pricing information
        
        Args:
            message: The AIMessage object containing tool calls, usage metadata, and response metadata
            response: The tool response/result to display
        """
        if not message.tool_calls:
            return
        
        with st.chat_message("assistant"):
            # Create columns: main content (wider) and cost info (narrower)
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.markdown("🛠️ **Tool Calls Executed:**")
                
                for tool_call in message.tool_calls:
                    tool_name = tool_call.get("name", "unknown")
                    tool_args = tool_call.get("args", {})
                    
                    with st.expander(f"🔧 {tool_name}", expanded=True):
                        if tool_args:
                            st.markdown("**Arguments:**")
                            st.json(tool_args)
                        
                        if response:
                            st.markdown("**Result:**")
                            st.code(response, language="json")
            
            with col2:
                # Display token usage and cost if available
                usage = message.usage_metadata
                if usage:
                    input_tokens = usage.get('input_tokens', 0)
                    output_tokens = usage.get('output_tokens', 0)
                    
                    # Extract model from response_metadata
                    response_metadata = message.response_metadata
                    model_name = response_metadata.get('model_name', '')
                    
                    # Calculate cost using just the model name
                    cost = PricingCalculator.calculate_cost(
                        model=model_name,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens
                    ) if model_name else None
                    
                    # Display cost information
                    if cost is not None:
                        formatted_cost = PricingCalculator.format_cost(cost)
                        st.markdown(f"**💰 {formatted_cost}**")
                    st.caption(f"🔼 {input_tokens:,} in")
                    st.caption(f"🔽 {output_tokens:,} out")
    
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