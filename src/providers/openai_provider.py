from typing import Generator, List, Tuple

from openai import OpenAI
from openai.types.chat import ChatCompletion

from .base import ChatMessage, LLMProvider, RawResponse, RawStreamChunk


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
        # Type narrowing: stream=False guarantees ChatCompletion
        assert isinstance(response, ChatCompletion)
        return response.choices[0].message.content or ""
    
    def stream(self, messages: List[ChatMessage], model: str) -> Generator[str, None, None]:
        """Streaming chat completion"""
        formatted = self.format_messages(messages)
        stream_response = self.client.chat.completions.create(
            model=model,
            messages=formatted,  # type: ignore[arg-type]
            stream=True
        )
        # Type narrowing: stream=True guarantees Stream
        assert not isinstance(stream_response, ChatCompletion)
        for event in stream_response:
            if event.choices[0].delta.content:
                yield event.choices[0].delta.content
    
    def complete_with_raw(self, messages: List[ChatMessage], model: str) -> Tuple[str, RawResponse]:
        """Non-streaming chat completion with raw response"""
        formatted = self.format_messages(messages)
        response = self.client.chat.completions.create(
            model=model,
            messages=formatted,  # type: ignore[arg-type]
            stream=False
        )
        # Type narrowing: stream=False guarantees ChatCompletion
        assert isinstance(response, ChatCompletion)
        content = response.choices[0].message.content or ""
        
        # Convert response to dict for JSON serialization
        raw_response: RawResponse = {
            "id": response.id,
            "object": response.object,
            "created": response.created,
            "model": response.model,
            "choices": [
                {
                    "index": choice.index,
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content,
                        "function_call": choice.message.function_call.model_dump() if choice.message.function_call else None,
                        "tool_calls": [tc.model_dump() for tc in choice.message.tool_calls] if choice.message.tool_calls else None,
                    },
                    "finish_reason": choice.finish_reason,
                    "logprobs": choice.logprobs.model_dump() if choice.logprobs else None,
                }
                for choice in response.choices
            ],
            "system_fingerprint": response.system_fingerprint,
        }
        
        # Add usage separately to match TypedDict structure
        if response.usage:
            raw_response["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        
        return content, raw_response
    
    def stream_with_raw(self, messages: List[ChatMessage], model: str) -> Generator[Tuple[str, RawStreamChunk], None, None]:
        """Streaming chat completion with raw chunks"""
        formatted = self.format_messages(messages)
        stream_response = self.client.chat.completions.create(
            model=model,
            messages=formatted,  # type: ignore[arg-type]
            stream=True
        )
        # Type narrowing: stream=True guarantees Stream
        assert not isinstance(stream_response, ChatCompletion)
        
        for event in stream_response:
            content = ""
            if event.choices and len(event.choices) > 0 and event.choices[0].delta.content:
                content = event.choices[0].delta.content
            
            # Convert chunk to dict
            raw_chunk: RawStreamChunk = {
                "id": event.id,
                "object": event.object,
                "created": event.created,
                "model": event.model,
                "choices": [
                    {
                        "index": choice.index,
                        "delta": {
                            "role": choice.delta.role,
                            "content": choice.delta.content,
                            "function_call": choice.delta.function_call.model_dump() if choice.delta.function_call else None,
                            "tool_calls": [tc.model_dump() for tc in choice.delta.tool_calls] if choice.delta.tool_calls else None,
                        } if choice.delta else {},
                        "finish_reason": choice.finish_reason,
                        "logprobs": choice.logprobs.model_dump() if choice.logprobs else None,
                    }
                    for choice in event.choices
                ] if event.choices else [],
                "system_fingerprint": event.system_fingerprint,
            }
            
            yield content, raw_chunk
