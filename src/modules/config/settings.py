import os
from dataclasses import dataclass
from typing import Dict

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ProviderConfig:
    """Configuration for a provider"""
    api_key_env: str
    default_model_env: str
    default_model: str


# Registry of available providers
PROVIDER_REGISTRY: Dict[str, ProviderConfig] = {
    "OpenAI": ProviderConfig(
        api_key_env="OPENAI_API_KEY",
        default_model_env="OPENAI_MODEL",
        default_model="gpt-4o-mini"
    ),
    "Mistral": ProviderConfig(
        api_key_env="MISTRAL_API_KEY",
        default_model_env="MISTRAL_MODEL",
        default_model="mistral-small-latest"
    ),
}


class Config:
    """Centralized configuration management"""
    
    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
    
    @staticmethod
    def get_provider_names() -> list[str]:
        """Get list of available provider names"""
        return list(PROVIDER_REGISTRY.keys())
    
    @staticmethod
    def get_default_model(provider_name: str) -> str:
        """Get default model for a provider"""
        config = PROVIDER_REGISTRY.get(provider_name)
        if not config:
            return ""
        return os.getenv(config.default_model_env, config.default_model)