"""
Pricing information for LLM models.

Prices are per 1 million tokens (USD).
Note: Prices are approximate and may change. Always verify with the provider's official pricing page.
"""

from typing import Dict, List, Optional, TypedDict


class ModelPricing(TypedDict):
    """Pricing structure for a model"""
    input_price: float  # Price per 1M input tokens
    output_price: float  # Price per 1M output tokens
    currency: str  # Currency code (e.g., "USD")
    provider: str  # Provider name (e.g., "OpenAI", "Mistral")


# Unified model pricing dictionary - model name is the key
# Models are organized by provider for documentation but stored in a flat structure
MODEL_PRICING: Dict[str, ModelPricing] = {
    # OpenAI Pricing (as of October 2024)
    # Source: https://openai.com/api/pricing/
    "gpt-4.1-2025-04-14": {
        "input_price": 2,
        "output_price": 8,
        "currency": "USD",
        "provider": "OpenAI"
    },
    "gpt-4.1-mini-2025-04-14": {
        "input_price": 0.4,
        "output_price": 1.6,
        "currency": "USD",
        "provider": "OpenAI"
    },
    "gpt-4.1-nano-2025-04-14": {
        "input_price": 0.1,
        "output_price": 0.4,
        "currency": "USD",
        "provider": "OpenAI"
    },
    
    # Mistral AI Pricing (as of October 2024)
    # Source: https://mistral.ai/technology/#pricing
    "mistral-medium-latest": {
        "input_price": 0.4,
        "output_price": 2.00,
        "currency": "USD",
        "provider": "Mistral"
    },
    "magistral-medium-latest": {
        "input_price": 2.00,
        "output_price": 5.00,
        "currency": "USD",
        "provider": "Mistral"
    },
    "codestral-latest": {
        "input_price": 0.3,
        "output_price": 0.9,
        "currency": "USD",
        "provider": "Mistral"
    },
}


class PricingCalculator:
    """Calculate costs for LLM API usage"""
    
    @staticmethod
    def get_model_pricing(model: str) -> Optional[ModelPricing]:
        """Get pricing information for a specific model
        
        Args:
            model: The model name (e.g., "gpt-4.1", "mistral-medium-latest")
            
        Returns:
            ModelPricing dictionary if found, None otherwise
        """
        return MODEL_PRICING.get(model)
    
    @staticmethod
    def get_available_models(provider: str) -> List[str]:
        """Get list of available models for a provider
        
        Args:
            provider: The provider name (e.g., "OpenAI", "Mistral")
            
        Returns:
            Sorted list of model names for the provider
        """
        models = [
            model_name 
            for model_name, pricing in MODEL_PRICING.items() 
            if pricing["provider"] == provider
        ]
        return sorted(models)
    
    @staticmethod
    def calculate_cost(
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> Optional[float]:
        """Calculate the cost of a request in USD
        
        Args:
            model: The model name (e.g., "gpt-4.1", "mistral-medium-latest")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Total cost in USD, or None if pricing not available
        """
        pricing = PricingCalculator.get_model_pricing(model)
        if not pricing:
            return None
        
        # Prices are per 1M tokens
        input_cost = (input_tokens / 1_000_000) * pricing["input_price"]
        output_cost = (output_tokens / 1_000_000) * pricing["output_price"]
        
        return input_cost + output_cost
    
    @staticmethod
    def format_cost(cost: float, currency: str = "USD") -> str:
        """Format cost for display"""
        if cost < 0.01:
            # For very small costs, show more decimal places
            return f"${cost:.6f}"
        elif cost < 1.0:
            return f"${cost:.4f}"
        else:
            return f"${cost:.2f}"
