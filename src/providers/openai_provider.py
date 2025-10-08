from typing import Generator, List, cast

from openai import OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from .base import ChatMessage, LLMProvider


class OpenAIProvider(LLMProvider[OpenAI]):
    """OpenAI-specific implementation"""
    
    def _initialize_client(self) -> OpenAI:
        if not self.api_key:
            raise RuntimeError("Missing OPENAI_API_KEY")
        return OpenAI(api_key=self.api_key)
    
    def complete(self, messages: List[ChatMessage], model: str) -> str:
        """Non-streaming chat completion"""
        formatted = self.format_messages(messages)
        response = self.client.chat.completions.create(
            model=model,
            messages=formatted,  # type: ignore[arg-type]
            stream=False
        )
        # Cast to ChatCompletion since stream=False guarantees non-streaming response
        # This narrows the Union[ChatCompletion, Stream[ChatCompletionChunk]] type
        response = cast(ChatCompletion, response)
        return response.choices[0].message.content or ""
    
    def stream(self, messages: List[ChatMessage], model: str) -> Generator[str, None, None]:
        """Streaming chat completion"""
        formatted = self.format_messages(messages)
        stream_response = self.client.chat.completions.create(
            model=model,
            messages=formatted,  # type: ignore[arg-type]
            stream=True
        )
        # Cast to Stream since stream=True guarantees streaming response
        # This narrows the Union[ChatCompletion, Stream[ChatCompletionChunk]] type
        stream_response = cast(Stream[ChatCompletionChunk], stream_response)
        for event in stream_response:
            if event.choices[0].delta.content:
                yield event.choices[0].delta.content
