"""
Test the compute_impact module.
"""
import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from modules.ecologits.compute_impact import (compute_generation_impact,
                                              format_impact_summary,
                                              get_model_parameters,
                                              get_server_location,
                                              load_electricity_mix,
                                              load_models)
from modules.ecologits.range_value import RangeValue


def test_load_electricity_mix():
    """Test loading electricity mix data."""
    mix = load_electricity_mix()
    
    assert 'NLD' in mix, "Netherlands should be in electricity mix"
    assert 'USA' in mix, "USA should be in electricity mix"
    
    # Check NLD has all required fields
    assert 'adpe' in mix['NLD']
    assert 'pe' in mix['NLD']
    assert 'gwp' in mix['NLD']
    assert 'wue' in mix['NLD']
    
    print("✓ Electricity mix loaded successfully")


def test_load_models():
    """Test loading model data."""
    models = load_models()
    
    # Check that models from the JSON are loaded
    assert 'codestral-latest' in models
    assert 'gpt-4.1' in models
    assert 'gpt-4.1-mini' in models
    assert 'mistral-medium-latest' in models
    
    # Check model structure
    model = models['codestral-latest']
    assert 'provider' in model
    assert 'architecture' in model
    assert model['provider'] == 'mistralai'
    
    print("✓ Models loaded successfully")


def test_get_server_location():
    """Test server location mapping."""
    assert get_server_location('mistralai') == 'NLD'
    assert get_server_location('openai') == 'USA'
    assert get_server_location('unknown') == 'USA'  # Default
    
    print("✓ Server location mapping works correctly")


def test_get_model_parameters():
    """Test parameter extraction."""
    models = load_models()
    
    # Test dense model with fixed parameters
    codestral = models['codestral-latest']
    active, total = get_model_parameters(codestral)
    assert active == 22.2
    assert total == 22.2
    
    # Test MoE model with ranges
    gpt4 = models['gpt-4.1']
    active, total = get_model_parameters(gpt4)
    assert isinstance(active, RangeValue)
    assert active.min == 35
    assert active.max == 106
    assert total == 352
    
    # Test model with parameter range
    mistral_medium = models['mistral-medium-latest']
    active, total = get_model_parameters(mistral_medium)
    assert isinstance(active, RangeValue)
    assert isinstance(total, RangeValue)
    assert active.min == 70
    assert active.max == 120
    
    print("✓ Model parameter extraction works correctly")


def test_compute_impact_codestral():
    """Test impact computation for Codestral."""
    impacts = compute_generation_impact(
        model_name="codestral-latest",
        input_tokens=100,
        output_tokens=500
    )
    
    # Check that all impact values are present and positive
    assert impacts.energy.value > 0
    assert impacts.gwp.value > 0
    assert impacts.adpe.value > 0
    assert impacts.pe.value > 0
    assert impacts.wcf.value > 0
    
    # Check units
    assert impacts.energy.unit == "kWh"
    assert impacts.gwp.unit == "kgCO2eq"
    assert impacts.adpe.unit == "kgSbeq"
    assert impacts.pe.unit == "MJ"
    assert impacts.wcf.unit == "L"
    
    print("✓ Codestral impact computation works correctly")


def test_compute_impact_gpt4():
    """Test impact computation for GPT-4.1."""
    impacts = compute_generation_impact(
        model_name="gpt-4.1",
        input_tokens=200,
        output_tokens=1000
    )
    
    # Check that values are RangeValue (due to parameter ranges)
    assert isinstance(impacts.energy.value, RangeValue)
    assert isinstance(impacts.gwp.value, RangeValue)
    
    # Check that ranges are valid
    assert impacts.energy.value.min > 0
    assert impacts.energy.value.max > impacts.energy.value.min
    
    print("✓ GPT-4.1 impact computation works correctly")


def test_format_impact_summary():
    """Test impact summary formatting."""
    impacts = compute_generation_impact(
        model_name="codestral-latest",
        input_tokens=100,
        output_tokens=500
    )
    
    summary = format_impact_summary(impacts)
    
    # Check that summary contains expected sections
    assert "Environmental Impact Summary" in summary
    assert "Energy:" in summary
    assert "GWP" in summary
    assert "Usage Phase:" in summary
    assert "Embodied Phase:" in summary
    
    print("✓ Impact summary formatting works correctly")


def test_invalid_model():
    """Test error handling for invalid model."""
    try:
        compute_generation_impact(
            model_name="nonexistent-model",
            input_tokens=100,
            output_tokens=500
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "not found" in str(e)
        print("✓ Invalid model error handling works correctly")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*50)
    print("Running tests for compute_impact module")
    print("="*50 + "\n")
    
    test_load_electricity_mix()
    test_load_models()
    test_get_server_location()
    test_get_model_parameters()
    test_compute_impact_codestral()
    test_compute_impact_gpt4()
    test_format_impact_summary()
    test_invalid_model()
    
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50 + "\n")


if __name__ == "__main__":
    run_all_tests()
