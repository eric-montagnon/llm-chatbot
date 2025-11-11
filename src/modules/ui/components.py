from typing import Generator, List, Optional, Tuple

import streamlit as st
from langchain.messages import AIMessage, HumanMessage, SystemMessage

from modules.config.pricing import PricingCalculator
from modules.config.settings import Config
from modules.ecologits.compute_impact import compute_generation_impact
from modules.ecologits.impacts.modeling import Impacts


def compute_impact_for_message(model_name: str, input_tokens: int, output_tokens: int) -> Optional[Impacts]:
    """Compute environmental impact for a message.
    
    Args:
        model_name: The model name from the API
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        Impacts object or None if model not supported
    """
    try:
        return compute_generation_impact(
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
    except (ValueError, KeyError):
        # Model not found in impact database
        return None


def format_impact_compact(impacts: Impacts) -> str:
    """Format impact as a compact string for display (always shows mean value).
    
    Args:
        impacts: Impacts object from compute_generation_impact
        
    Returns:
        Compact formatted string with mean values
    """
    # Helper function to get mean value
    def get_mean_value(val):
        if hasattr(val, 'mean'):
            # It's a RangeValue, use mean
            return val.mean
        else:
            # It's a simple value
            return val
    
    # Format energy in mWh for readability
    energy_kwh = get_mean_value(impacts.energy.value)
    energy_mwh = energy_kwh * 1000  # Convert kWh to mWh
    
    # Format GWP in g for readability
    gwp_kg = get_mean_value(impacts.gwp.value)
    gwp_g = gwp_kg * 1000  # Convert kg to g
    
    # Format water in mL for readability
    water_l = get_mean_value(impacts.wcf.value)
    water_ml = water_l * 1000  # Convert L to mL
    
    return f"⚡ {energy_mwh:.2f} mWh | 🌍 {gwp_g:.2f} g CO₂eq | 💧 {water_ml:.2f} mL"


def calculate_total_cost(messages: List[AIMessage | HumanMessage | SystemMessage]) -> float:
    """Calculate total cost from all AI messages in the conversation.
    
    Args:
        messages: List of conversation messages
        
    Returns:
        Total cost in USD
    """
    total_cost = 0.0
    
    for message in messages:
        if isinstance(message, AIMessage):
            usage = message.usage_metadata
            if usage:
                input_tokens = usage.get('input_tokens', 0)
                output_tokens = usage.get('output_tokens', 0)
                response_metadata = message.response_metadata
                model_name = response_metadata.get('model_name', '')
                
                if model_name:
                    cost = PricingCalculator.calculate_cost(
                        model=model_name,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens
                    )
                    if cost is not None:
                        total_cost += cost
    
    return total_cost


def calculate_total_impact(messages: List[AIMessage | HumanMessage | SystemMessage]) -> Optional[dict]:
    """Calculate total environmental impact from all AI messages in the conversation.
    
    Args:
        messages: List of conversation messages
        
    Returns:
        Dictionary with total energy (mWh), gwp (g CO2eq), and water (mL), or None if no impact data
    """
    total_energy_kwh = 0.0
    total_gwp_kg = 0.0
    total_water_l = 0.0
    has_impact = False
    
    for message in messages:
        if isinstance(message, AIMessage):
            usage = message.usage_metadata
            if usage:
                input_tokens = usage.get('input_tokens', 0)
                output_tokens = usage.get('output_tokens', 0)
                response_metadata = message.response_metadata
                model_name = response_metadata.get('model_name', '')
                
                if model_name:
                    impacts = compute_impact_for_message(model_name, input_tokens, output_tokens)
                    if impacts:
                        has_impact = True
                        # Helper function to get mean value
                        def get_mean_value(val):
                            if hasattr(val, 'mean'):
                                return val.mean
                            else:
                                return val
                        
                        total_energy_kwh += get_mean_value(impacts.energy.value)
                        total_gwp_kg += get_mean_value(impacts.gwp.value)
                        total_water_l += get_mean_value(impacts.wcf.value)
    
    if not has_impact:
        return None
    
    return {
        'energy_mwh': total_energy_kwh * 1000,  # Convert kWh to mWh
        'gwp_g': total_gwp_kg * 1000,  # Convert kg to g
        'water_ml': total_water_l * 1000  # Convert L to mL
    }



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
            # Create columns: main content (wider), cost info, and ecological impact
            col1, col2, col3 = st.columns([4, 1, 1])
            
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
            
            with col3:
                # Display environmental impact in third column
                usage = message.usage_metadata
                if usage:
                    input_tokens = usage.get('input_tokens', 0)
                    output_tokens = usage.get('output_tokens', 0)
                    response_metadata = message.response_metadata
                    model_name = response_metadata.get('model_name', '')
                    
                    if model_name:
                        impacts = compute_impact_for_message(model_name, input_tokens, output_tokens)
                        if impacts:
                            st.markdown("**🌍 Impact**")
                            impact_str = format_impact_compact(impacts)
                            st.caption(impact_str)
                        else:
                            st.caption("_Impact data not available_")
    
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
            # Create columns: main content (wider), cost info, and ecological impact
            col1, col2, col3 = st.columns([4, 1, 1])
            
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
            
            with col3:
                # Display environmental impact in third column
                usage = message.usage_metadata
                if usage:
                    input_tokens = usage.get('input_tokens', 0)
                    output_tokens = usage.get('output_tokens', 0)
                    response_metadata = message.response_metadata
                    model_name = response_metadata.get('model_name', '')
                    
                    if model_name:
                        impacts = compute_impact_for_message(model_name, input_tokens, output_tokens)
                        if impacts:
                            st.markdown("**🌍 Impact**")
                            impact_str = format_impact_compact(impacts)
                            st.caption(impact_str)
                        else:
                            st.caption("_Impact data not available_")
    
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