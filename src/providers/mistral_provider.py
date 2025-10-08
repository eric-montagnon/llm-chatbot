from typing import Generator, List

from mistralai import Mistral

from .base import ChatMessage, LLMProvider


class MistralProvider(LLMProvider):
    
    def _initialize_client(self) -> Mistral:
        if not self.api_key:
            raise RuntimeError("Missing MISTRAL_API_KEY")
        return Mistral(api_key=self.api_key)
    
    @property
    def client(self) -> Mistral:
        """Lazy initialization of client"""
        if self._client is None:
            self._client = self._initialize_client()
        # Type narrowing: we know _initialize_client returns Mistral
        if not isinstance(self._client, Mistral):
            raise TypeError("Expected Mistral client")
        return self._client
    
    def complete(self, messages: List[ChatMessage], model: str) -> str:
        """Non-streaming chat completion"""
        formatted = self.format_messages(messages)
        response = self.client.chat.complete(
            model=model,
            messages=formatted  # type: ignore[arg-type]
        )
        message = response.choices[0].message
        if isinstance(message, dict):
            content = message.get("content", "")
            return content if isinstance(content, str) else ""
        content = getattr(message, "content", None)
        return content if isinstance(content, str) else ""
    
    def stream(self, messages: List[ChatMessage], model: str) -> Generator[str, None, None]:
        """Streaming chat completion"""
        formatted = self.format_messages(messages)
        stream = self.client.chat.stream(
            model=model,
            messages=formatted  # type: ignore[arg-type]
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