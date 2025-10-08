from typing import Any, Dict, Generator, List, Tuple, cast

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
    
    def complete_with_raw(self, messages: List[ChatMessage], model: str) -> Tuple[str, Dict[str, Any]]:
        """Non-streaming chat completion with raw response"""
        formatted = self.format_messages(messages)
        response = self.client.chat.completions.create(
            model=model,
            messages=formatted,  # type: ignore[arg-type]
            stream=False
        )
        response = cast(ChatCompletion, response)
        content = response.choices[0].message.content or ""
        
        # Convert response to dict for JSON serialization
        raw_response = {
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
                    "logprobs": choice.logprobs,
                }
                for choice in response.choices
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
                "total_tokens": response.usage.total_tokens if response.usage else None,
            } if response.usage else None,
            "system_fingerprint": response.system_fingerprint,
        }
        
        return content, raw_response
    
    def stream_with_raw(self, messages: List[ChatMessage], model: str) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """Streaming chat completion with raw chunks"""
        formatted = self.format_messages(messages)
        stream_response = self.client.chat.completions.create(
            model=model,
            messages=formatted,  # type: ignore[arg-type]
            stream=True
        )
        stream_response = cast(Stream[ChatCompletionChunk], stream_response)
        
        for event in stream_response:
            content = ""
            if event.choices and len(event.choices) > 0 and event.choices[0].delta.content:
                content = event.choices[0].delta.content
            
            # Convert chunk to dict
            raw_chunk = {
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
                        "logprobs": choice.logprobs,
                    }
                    for choice in event.choices
                ] if event.choices else [],
                "system_fingerprint": event.system_fingerprint,
            }
            
            yield content, raw_chunk
