"""
Compute environmental impact of LLM generation based on model, input tokens, and output tokens.
"""
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add parent directories to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent
sys.path.insert(0, str(src_dir))

from modules.ecologits.impacts.llm import compute_llm_impacts
from modules.ecologits.impacts.modeling import Impacts
from modules.ecologits.range_value import RangeValue, ValueOrRange

# Data center PUE (Power Usage Effectiveness) values
# Source: Typical values for modern data centers
DATACENTER_PUE = 1.2

# Data center WUE (Water Usage Effectiveness) values in L/kWh
# Source: Typical values for modern data centers
DATACENTER_WUE = 1.8


def load_electricity_mix() -> Dict[str, Dict[str, float]]:
    """
    Load electricity mix data from CSV file.
    
    Returns:
        Dictionary with country codes as keys and impact factors as values.
    """
    electricity_mix = {}
    csv_path = Path(__file__).parent / "data" / "electricity_mixes.csv"
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            country = row['name']
            electricity_mix[country] = {
                'adpe': float(row['adpe']),
                'pe': float(row['pe']),
                'gwp': float(row['gwp']),
                'wue': float(row['wue'])
            }
    
    return electricity_mix


def load_models() -> Dict[str, Any]:
    """
    Load model data from JSON file.
    
    Returns:
        Dictionary with model names as keys and model data as values.
    """
    models = {}
    json_path = Path(__file__).parent / "data" / "models.json"
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        for model in data['models']:
            models[model['name']] = model
    
    return models


def get_model_parameters(model_data: Dict[str, Any]) -> tuple[ValueOrRange, ValueOrRange]:
    """
    Extract active and total parameters from model data.
    
    Args:
        model_data: Model data dictionary
        
    Returns:
        Tuple of (active_parameters, total_parameters) in billions
    """
    arch = model_data['architecture']
    params = arch['parameters']
    
    # Handle different parameter formats
    if isinstance(params, (int, float)):
        # Dense model with fixed parameters
        return params, params
    elif isinstance(params, dict):
        if 'total' in params and 'active' in params:
            # MoE model with total and active parameters
            total = params['total']
            active = params['active']
            
            if isinstance(active, dict) and 'min' in active and 'max' in active:
                active_range = RangeValue(min=active['min'], max=active['max'])
            else:
                active_range = active
                
            return active_range, total
        elif 'min' in params and 'max' in params:
            # Dense model with parameter range
            param_range = RangeValue(min=params['min'], max=params['max'])
            return param_range, param_range
    
    raise ValueError(f"Unknown parameter format: {params}")


def get_server_location(provider: str) -> str:
    """
    Get server location (country code) based on provider.
    
    Args:
        provider: Model provider name
        
    Returns:
        Country code (NLD for Netherlands, USA for United States)
    """
    location_map = {
        'mistralai': 'NLD',  # Netherlands
        'openai': 'USA',     # United States
    }
    
    return location_map.get(provider, 'USA')  # Default to USA


def compute_generation_impact(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    request_latency: Optional[float] = None
) -> Impacts:
    """
    Compute environmental impact of an LLM generation request.
    
    Args:
        model_name: Name of the model (e.g., "gpt-4.1", "codestral-latest")
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        request_latency: Optional measured request latency in seconds
        
    Returns:
        Impacts object containing all environmental impact metrics
        
    Raises:
        ValueError: If model is not found in the database
    """
    # Load data
    models = load_models()
    electricity_mix = load_electricity_mix()
    
    # Get model data
    if model_name not in models:
        available_models = ', '.join(models.keys())
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {available_models}"
        )
    
    model_data = models[model_name]
    
    # Get model parameters
    active_params, total_params = get_model_parameters(model_data)
    
    # Get server location and electricity mix
    provider = model_data['provider']
    location = get_server_location(provider)
    mix = electricity_mix[location]
    
    # Compute impacts
    impacts = compute_llm_impacts(
        model_active_parameter_count=active_params,
        model_total_parameter_count=total_params,
        output_token_count=float(output_tokens),
        request_latency=0.1,
        if_electricity_mix_adpe=mix['adpe'],
        if_electricity_mix_pe=mix['pe'],
        if_electricity_mix_gwp=mix['gwp'],
        if_electricity_mix_wue=mix['wue'],
        datacenter_pue=DATACENTER_PUE,
        datacenter_wue=DATACENTER_WUE
    )
    
    return impacts


def format_impact_summary(impacts: Impacts) -> str:
    """
    Format impact results as a human-readable summary.
    
    Args:
        impacts: Impacts object from compute_generation_impact
        
    Returns:
        Formatted string with impact summary
    """
    lines = [
        "Environmental Impact Summary",
        "=" * 50,
        f"Energy: {impacts.energy.value:.6f} {impacts.energy.unit}",
        f"GWP (CO2 eq): {impacts.gwp.value:.6f} {impacts.gwp.unit}",
        f"ADPe (Sb eq): {impacts.adpe.value:.9f} {impacts.adpe.unit}",
        f"PE (Primary Energy): {impacts.pe.value:.6f} {impacts.pe.unit}",
        f"Water: {impacts.wcf.value:.6f} {impacts.wcf.unit}",
        "",
        "Usage Phase:",
        f"  Energy: {impacts.usage.energy.value:.6f} {impacts.usage.energy.unit}",
        f"  GWP: {impacts.usage.gwp.value:.6f} {impacts.usage.gwp.unit}",
        "",
        "Embodied Phase:",
        f"  GWP: {impacts.embodied.gwp.value:.9f} {impacts.embodied.gwp.unit}",
    ]
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    print("Example 1: GPT-4.1")
    print("-" * 50)
    impacts = compute_generation_impact(
        model_name="gpt-4.1",
        input_tokens=100,
        output_tokens=500
    )
    print(format_impact_summary(impacts))
    print()
    
    print("Example 2: Codestral")
    print("-" * 50)
    impacts = compute_generation_impact(
        model_name="codestral-latest",
        input_tokens=200,
        output_tokens=1000
    )
    print(format_impact_summary(impacts))
