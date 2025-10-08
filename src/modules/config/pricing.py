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


# OpenAI Pricing (as of October 2024)
# Source: https://openai.com/api/pricing/
OPENAI_PRICING: Dict[str, ModelPricing] = {
    "gpt-4.1": {
        "input_price": 2,
        "output_price": 8,
        "currency": "USD"
    },
    "gpt-4.1-mini": {
        "input_price": 0.4,
        "output_price": 1.6,
        "currency": "USD"
    },
    "gpt-4.1-nano": {
        "input_price": 0.1,
        "output_price": 0.4,
        "currency": "USD"
    },
}

# Mistral AI Pricing (as of October 2024)
# Source: https://mistral.ai/technology/#pricing
MISTRAL_PRICING: Dict[str, ModelPricing] = {
    "mistral-medium-latest": {
        "input_price": 0.4,
        "output_price": 2.00,
        "currency": "USD"
    },
    "magistral-medium-latest": {
        "input_price": 2.00,
        "output_price": 5.00,
        "currency": "USD"
    },
    "codestral-latest": {
        "input_price": 0.3,
        "output_price": 0.9,
        "currency": "USD"
    },
}

# Provider pricing registry
PROVIDER_PRICING: Dict[str, Dict[str, ModelPricing]] = {
    "OpenAI": OPENAI_PRICING,
    "Mistral": MISTRAL_PRICING,
}


class PricingCalculator:
    """Calculate costs for LLM API usage"""
    
    @staticmethod
    def get_model_pricing(provider: str, model: str) -> Optional[ModelPricing]:
        """Get pricing information for a specific model"""
        provider_prices = PROVIDER_PRICING.get(provider, {})
        return provider_prices.get(model)
    
    @staticmethod
    def get_available_models(provider: str) -> List[str]:
        """Get list of available models for a provider"""
        provider_prices = PROVIDER_PRICING.get(provider, {})
        return sorted(provider_prices.keys())
    
    @staticmethod
    def calculate_cost(
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> Optional[float]:
        """Calculate the cost of a request in USD"""
        pricing = PricingCalculator.get_model_pricing(provider, model)
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
