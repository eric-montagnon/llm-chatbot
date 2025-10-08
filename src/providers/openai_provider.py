from typing import List, Generator
from openai import OpenAI
from .base import LLMProvider, ChatMessage


class OpenAIProvider(LLMProvider):
    """OpenAI-specific implementation"""
    
    def _initialize_client(self):
        if not self.api_key:
            raise RuntimeError("Missing OPENAI_API_KEY")
        return OpenAI(api_key=self.api_key)
    
    def complete(self, messages: List[ChatMessage], model: str) -> str:
        """Non-streaming chat completion"""
        formatted = self.format_messages(messages)
        response = self.client.chat.completions.create(
            model=model,
            messages=formatted,
            stream=False
        )
        return response.choices[0].message.content or ""
    
    def stream(self, messages: List[ChatMessage], model: str) -> Generator[str, None, None]:
        """Streaming chat completion"""
        formatted = self.format_messages(messages)
        stream = self.client.chat.completions.create(
            model=model,
            messages=formatted,
            stream=True
        )
        for event in stream:
            if event.choices[0].delta.content:
                yield event.choices[0].delta.content