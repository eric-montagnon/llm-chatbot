from typing import Any, Dict, Generator, List, Tuple

from mistralai import Mistral

from .base import ChatMessage, LLMProvider


class MistralProvider(LLMProvider[Mistral]):
    
    def _initialize_client(self) -> Mistral:
        if not self.api_key:
            raise RuntimeError("Missing MISTRAL_API_KEY")
        return Mistral(api_key=self.api_key)
    
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
    
    def complete_with_raw(self, messages: List[ChatMessage], model: str) -> Tuple[str, Dict[str, Any]]:
        """Non-streaming chat completion with raw response"""
        formatted = self.format_messages(messages)
        response = self.client.chat.complete(
            model=model,
            messages=formatted  # type: ignore[arg-type]
        )
        
        # Extract content
        message = response.choices[0].message
        if isinstance(message, dict):
            content = message.get("content", "")
            content = content if isinstance(content, str) else ""
        else:
            content = getattr(message, "content", None)
            content = content if isinstance(content, str) else ""
        
        # Build raw response dict
        raw_response = {
            "id": getattr(response, "id", None),
            "object": getattr(response, "object", "chat.completion"),
            "created": getattr(response, "created", None),
            "model": getattr(response, "model", model),
            "choices": [],
            "usage": None
        }
        
        # Add choices
        if hasattr(response, "choices") and response.choices:
            for choice in response.choices:
                choice_dict = {
                    "index": getattr(choice, "index", 0),
                    "message": {},
                    "finish_reason": getattr(choice, "finish_reason", None)
                }
                
                # Extract message content
                msg = choice.message if hasattr(choice, "message") else choice
                if isinstance(msg, dict):
                    choice_dict["message"] = msg
                else:
                    choice_dict["message"] = {
                        "role": getattr(msg, "role", "assistant"),
                        "content": getattr(msg, "content", "")
                    }
                
                raw_response["choices"].append(choice_dict)
        
        # Add usage if available
        if hasattr(response, "usage"):
            usage = response.usage
            if isinstance(usage, dict):
                raw_response["usage"] = usage
            else:
                raw_response["usage"] = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None)
                }
        
        return content, raw_response
    
    def stream_with_raw(self, messages: List[ChatMessage], model: str) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """Streaming chat completion with raw chunks"""
        formatted = self.format_messages(messages)
        stream = self.client.chat.stream(
            model=model,
            messages=formatted  # type: ignore[arg-type]
        )
        
        for event in stream:
            content = ""
            raw_chunk = {
                "object": "chat.completion.chunk",
                "model": model,
                "choices": []
            }
            
            if hasattr(event, "data") and event.data:
                # Add event metadata
                raw_chunk["id"] = getattr(event.data, "id", None)
                raw_chunk["created"] = getattr(event.data, "created", None)
                
                if hasattr(event.data, "choices") and event.data.choices:
                    for choice in event.data.choices:
                        delta = choice.delta if hasattr(choice, "delta") else choice
                        
                        choice_dict = {
                            "index": getattr(choice, "index", 0),
                            "delta": {},
                            "finish_reason": getattr(choice, "finish_reason", None)
                        }
                        
                        # Extract delta content
                        if isinstance(delta, dict):
                            choice_dict["delta"] = delta
                            content = delta.get("content", "")
                        else:
                            delta_content = getattr(delta, "content", None)
                            if delta_content:
                                content = delta_content
                            choice_dict["delta"] = {
                                "role": getattr(delta, "role", None),
                                "content": delta_content
                            }
                        
                        raw_chunk["choices"].append(choice_dict)
            
            if content:
                yield content, raw_chunk
