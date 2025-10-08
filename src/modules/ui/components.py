from datetime import datetime
from typing import Generator, List, Tuple

import streamlit as st

from modules.chat.types import Interaction
from modules.config.pricing import PricingCalculator
from modules.config.settings import Config
from modules.providers.types import EcologicalImpact, ToolResultInfo


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
    def display_tool_calls(tool_results: List[ToolResultInfo], in_chat_context: bool = True):
        """Display tool call executions in a visually distinct way
        
        Args:
            tool_results: List of tool execution results
            in_chat_context: If True, wrap in st.chat_message("assistant"). 
                           If False, assumes already in a chat message context.
        """
        if not tool_results:
            return
        
        def render_tools():
            st.markdown("🛠️ **Tool Calls Executed:**")
            
            for tool_result in tool_results:
                tool_name = tool_result.get("name", "unknown")
                tool_args = tool_result.get("arguments", {})
                tool_output = tool_result.get("result", "")
                
                with st.expander(f"🔧 {tool_name}", expanded=True):
                    # Display arguments
                    if tool_args:
                        st.markdown("**Arguments:**")
                        st.json(tool_args)
                    
                    # Display result
                    st.markdown("**Result:**")
                    st.code(tool_output, language="json")
        
        if in_chat_context:
            with st.chat_message("assistant"):
                render_tools()
        else:
            render_tools()
    
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
            elif role == "tool":
                # Display tool result messages
                with st.expander(f"🛠️ Tool Result", expanded=True):
                    st.text(content)
        
        # Display tool execution results if available
        RawMessageViewer._display_tool_results(interaction)
        
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
            
            # Display ecological impact for non-streaming requests
            impact = raw.get("impact") if raw else None
            if impact is not None:
                RawMessageViewer._display_ecological_impact(impact)
                        
        elif interaction["type"] == "streaming":
            # For streaming, we don't have usage info typically
            st.info("ℹ️ Token usage information is not available for streaming responses")
    
    @staticmethod
    def _display_tool_results(interaction: Interaction):
        """Display tool execution results if available"""
        from modules.chat.types import (CompleteInteraction,
                                        StreamingInteraction)
        
        tool_results = None
        
        if interaction["type"] == "complete":
            # Type narrowing for CompleteInteraction
            complete_interaction: CompleteInteraction = interaction  # type: ignore[assignment]
            complete_response = complete_interaction["response"]
            raw = complete_response.get("raw")
            if raw:
                tool_results = raw.get("tool_results")
        elif interaction["type"] == "streaming":
            # Type narrowing for StreamingInteraction
            streaming_interaction: StreamingInteraction = interaction  # type: ignore[assignment]
            streaming_response = streaming_interaction["response"]
            chunks = streaming_response.get("chunks")
            if chunks and isinstance(chunks, list):
                for chunk in chunks:
                    chunk_tool_results = chunk.get("tool_results")
                    if chunk_tool_results:
                        tool_results = chunk_tool_results
                        break
        
        if tool_results:
            for idx, tool_result in enumerate(tool_results, 1):
                tool_name = tool_result.get("name", "unknown")
                tool_args = tool_result.get("arguments", {})
                tool_output = tool_result.get("result", "")
                
                with st.expander(f"Tool #{idx}: {tool_name}", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Arguments:**")
                        st.json(tool_args)
                    
                    with col2:
                        st.markdown("**Result:**")
                        st.code(tool_output, language="json")
    
    @staticmethod
    def _display_ecological_impact(impact: EcologicalImpact) -> None:
        """Display ecological impact metrics"""
        st.write("### 🌍 Ecological Impact")
        
        col1, col2 = st.columns(2)
        
        with col1:
            energy = impact.get("energy_kwh", 0)*1000  # Convert kWh to Wh
            st.metric(
                "Energy Consumption",
                f"{energy:.1f} Wh",
                help="Total energy consumed in watt-hours"
            )

            gwp = impact.get("gwp_kgco2eq", 0)*1000  # Convert kg to g
            st.metric(
                "Carbon Footprint (GWP)",
                f"{gwp:.1f} g CO₂eq",
                help="Global Warming Potential in g of CO₂ equivalent"
            )
        
        with col2:
            adpe = impact.get("adpe_kgsbeq", 0) * 1000  # Convert kg to g
            st.metric(
                "Abiotic Depletion (ADPe)",
                f"{adpe:.1f} g Sb eq",
                help="Abiotic Depletion Potential for elements in g of Antimony equivalent"
            )

            pe = impact.get("pe_mj", 0) * 1000000  # Convert MJ to J
            st.metric(
                "Primary Energy (PE)",
                f"{pe:.1f} J",
                help="Primary Energy consumption in Joules"
            )
    
    @staticmethod
    def display_streaming_status(chunk_count: int):
        """Display real-time streaming status"""
        return st.empty().info(f"🔄 Streaming... ({chunk_count} chunks received)")
