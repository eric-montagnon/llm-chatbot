"""
Core type definitions shared across the application.

This module contains type aliases and protocols that are used across multiple
modules to avoid circular imports while maintaining type safety.
"""

from typing import TYPE_CHECKING, Union

# Forward references to avoid circular imports
if TYPE_CHECKING:
    from modules.providers.mistral_provider import MistralProvider
    from modules.providers.openai_provider import OpenAIProvider


# ============================================================================
# Provider Type Alias
# ============================================================================

# Type alias for any LLM provider instance
# This is defined here to avoid circular imports while maintaining type safety
if TYPE_CHECKING:
    LLMProviderInstance = Union["OpenAIProvider", "MistralProvider"]
else:
    # At runtime, we don't need the specific types
    from typing import Protocol
    
    class LLMProviderInstance(Protocol):
        """Protocol for LLM provider instances"""
        def complete_with_raw(self, messages: list, model: str): ...
        def stream_with_raw(self, messages: list, model: str): ...


__all__ = ["LLMProviderInstance"]
