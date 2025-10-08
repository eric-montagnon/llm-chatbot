from typing import Generator, List

from mistralai import Mistral

from .base import ChatMessage, LLMProvider


class MistralProvider(LLMProvider):
    
    def _initialize_client(self):
        if not self.api_key:
            raise RuntimeError("Missing MISTRAL_API_KEY")
        return Mistral(api_key=self.api_key)
    
    def complete(self, messages: List[ChatMessage], model: str) -> str:
        """Non-streaming chat completion"""
        formatted = self.format_messages(messages)
        response = self.client.chat.complete(
            model=model,
            messages=formatted
        )
        message = response.choices[0].message
        if isinstance(message, dict):
            return message.get("content", "")
        return message.content or ""
    
    def stream(self, messages: List[ChatMessage], model: str) -> Generator[str, None, None]:
        """Streaming chat completion"""
        formatted = self.format_messages(messages)
        stream = self.client.chat.stream(
            model=model,
            messages=formatted
        )
        for event in stream:
            if hasattr(event, "data") and event.data and event.data.choices:
                delta = event.data.choices[0].delta
                if isinstance(delta, dict):
                    content = delta.get("content")
                else:
                    content = getattr(delta, "content", None)
                if content:
                    yield content