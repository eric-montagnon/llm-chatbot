from typing import Generator, List

from openai import OpenAI

from .base import ChatMessage, LLMProvider


class OpenAIProvider(LLMProvider):
    """OpenAI-specific implementation"""
    
    def _initialize_client(self) -> OpenAI:
        if not self.api_key:
            raise RuntimeError("Missing OPENAI_API_KEY")
        return OpenAI(api_key=self.api_key)
    
    @property
    def client(self) -> OpenAI:
        """Lazy initialization of client"""
        if self._client is None:
            self._client = self._initialize_client()
        if not isinstance(self._client, OpenAI):
            raise TypeError("Expected OpenAI client")
        return self._client
    
    def complete(self, messages: List[ChatMessage], model: str) -> str:
        """Non-streaming chat completion"""
        formatted = self.format_messages(messages)
        response = self.client.chat.completions.create(
            model=model,
            messages=formatted,  # type: ignore[arg-type]
            stream=False
        )
        return response.choices[0].message.content or ""
    
    def stream(self, messages: List[ChatMessage], model: str) -> Generator[str, None, None]:
        """Streaming chat completion"""
        formatted = self.format_messages(messages)
        stream_response = self.client.chat.completions.create(
            model=model,
            messages=formatted,  # type: ignore[arg-type]
            stream=True
        )
        for event in stream_response:
            if event.choices[0].delta.content:
                yield event.choices[0].delta.content
