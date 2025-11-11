"""
Integration example: Add environmental impact tracking to chatbot responses.

This module shows how to integrate the environmental impact calculator
with the LLM chatbot to display the carbon footprint of each generation.
"""
import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from modules.ecologits.compute_impact import (compute_generation_impact)


def calculate_and_display_impact(model_name: str, prompt_tokens: int, completion_tokens: int):
    """
    Calculate and display environmental impact for a chatbot response.
    
    Args:
        model_name: Name of the model used
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
    """
    try:
        # Compute the environmental impact
        impacts = compute_generation_impact(
            model_name=model_name,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens
        )
        
        # Format a concise summary
        energy_kwh = impacts.energy.value
        gwp_kg_co2 = impacts.gwp.value
        water_liters = impacts.wcf.value
        
        # Convert to more readable units if needed
        if hasattr(energy_kwh, 'mean'):
            energy_kwh = energy_kwh.mean
        if hasattr(gwp_kg_co2, 'mean'):
            gwp_kg_co2 = gwp_kg_co2.mean
        if hasattr(water_liters, 'mean'):
            water_liters = water_liters.mean
        
        # Convert kgCO2 to gCO2 for readability
        gwp_g_co2 = gwp_kg_co2 * 1000
        
        return {
            'energy_kwh': energy_kwh,
            'gwp_g_co2': gwp_g_co2,
            'water_liters': water_liters,
            'summary': (
                f"🌍 Environmental Impact: "
                f"{gwp_g_co2:.2f}g CO₂ | "
                f"{energy_kwh*1000:.2f}Wh | "
                f"{water_liters:.2f}L water"
            )
        }
    except ValueError as e:
        # Model not found or other error
        return {
            'error': str(e),
            'summary': f"⚠️ Could not calculate impact: {e}"
        }


# Example integration with OpenAI-style response
def example_openai_integration():
    """Example showing how to integrate with OpenAI response."""
    
    # Simulated OpenAI response
    response = {
        'model': 'gpt-4.1',
        'usage': {
            'prompt_tokens': 150,
            'completion_tokens': 450,
            'total_tokens': 600
        },
        'choices': [{
            'message': {
                'content': 'This is the AI response...'
            }
        }]
    }
    
    # Calculate impact
    impact = calculate_and_display_impact(
        model_name=response['model'],
        prompt_tokens=response['usage']['prompt_tokens'],
        completion_tokens=response['usage']['completion_tokens']
    )
    
    print("AI Response:", response['choices'][0]['message']['content'])
    print(impact['summary'])
    print()


# Example integration with Mistral-style response
def example_mistral_integration():
    """Example showing how to integrate with Mistral response."""
    
    # Simulated Mistral response
    response = {
        'model': 'codestral-latest',
        'usage': {
            'prompt_tokens': 200,
            'completion_tokens': 800,
            'total_tokens': 1000
        },
        'choices': [{
            'message': {
                'content': 'Here is the code you requested...'
            }
        }]
    }
    
    # Calculate impact
    impact = calculate_and_display_impact(
        model_name=response['model'],
        prompt_tokens=response['usage']['prompt_tokens'],
        completion_tokens=response['usage']['completion_tokens']
    )
    
    print("AI Response:", response['choices'][0]['message']['content'])
    print(impact['summary'])
    print()


# Example for Streamlit integration
def streamlit_integration_example():
    """Example showing how to display impact in Streamlit."""
    
    example_code = '''
import streamlit as st
from modules.ecologits.integration_example import calculate_and_display_impact

# After getting a response from the LLM
if response:
    # Extract usage information
    model_name = response.model
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    
    # Calculate impact
    impact = calculate_and_display_impact(
        model_name=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens
    )
    
    # Display in sidebar or at bottom of chat
    with st.sidebar:
        st.caption(impact['summary'])
    
    # Or show detailed metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("CO₂", f"{impact['gwp_g_co2']:.2f}g")
    with col2:
        st.metric("Energy", f"{impact['energy_kwh']*1000:.2f}Wh")
    with col3:
        st.metric("Water", f"{impact['water_liters']:.2f}L")
'''
    
    print("Streamlit Integration Example:")
    print("=" * 60)
    print(example_code)
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Environmental Impact Integration Examples")
    print("=" * 60 + "\n")
    
    print("Example 1: OpenAI Integration")
    print("-" * 60)
    example_openai_integration()
    
    print("Example 2: Mistral Integration")
    print("-" * 60)
    example_mistral_integration()
    
    print("Example 3: Streamlit Integration")
    print("-" * 60)
    streamlit_integration_example()
    
    print("=" * 60)
    print("Integration complete! You can now track environmental impact.")
    print("=" * 60 + "\n")
