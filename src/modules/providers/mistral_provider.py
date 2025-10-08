from typing import Generator, List, Tuple

from ecologits import EcoLogits
from mistralai import Mistral

from .base import LLMProvider
from .types import (ChatMessage, ChoiceInfo, EcologicalImpact, MessageContent,
                    RawResponse, RawStreamChunk, StreamChoiceInfo, UsageInfo)


class MistralProvider(LLMProvider[Mistral]):
    
    def _initialize_client(self) -> Mistral:
        if not self.api_key:
            raise RuntimeError("Missing MISTRAL_API_KEY")
        
        # Initialize EcoLogits tracer
        EcoLogits.init()
        
        return Mistral(api_key=self.api_key)
    
    def _extract_impacts(self, response: object) -> EcologicalImpact:
        """Extract ecological impacts from response"""
        if hasattr(response, 'impacts'):
            impacts = response.impacts  # type: ignore[attr-defined]
            return {
                "energy_kwh": impacts.energy.value,
                "gwp_kgco2eq": impacts.gwp.value,
                "adpe_kgsbeq": impacts.adpe.value,
                "pe_mj": impacts.pe.value,
            }
        # Return default values if no impacts available
        return {
            "energy_kwh": 0.0,
            "gwp_kgco2eq": 0.0,
            "adpe_kgsbeq": 0.0,
            "pe_mj": 0.0,
        }
    
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
    
    def complete_with_raw(self, messages: List[ChatMessage], model: str) -> Tuple[str, RawResponse]:
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
        raw_response: RawResponse = {
            "id": getattr(response, "id", None),
            "object": getattr(response, "object", "chat.completion"),
            "created": getattr(response, "created", None),
            "model": getattr(response, "model", model),
            "choices": [],
        }
        
        # Add choices
        choices_list: List[ChoiceInfo] = []
        if hasattr(response, "choices") and response.choices:
            for choice in response.choices:
                choice_dict: ChoiceInfo = {
                    "index": getattr(choice, "index", 0),
                    "message": {},
                    "finish_reason": getattr(choice, "finish_reason", None)
                }
                
                # Extract message content
                msg = choice.message if hasattr(choice, "message") else choice
                msg_content: MessageContent
                if isinstance(msg, dict):
                    msg_content = {
                        "role": msg.get("role"),
                        "content": msg.get("content"),
                        "function_call": msg.get("function_call"),
                        "tool_calls": msg.get("tool_calls"),
                    }
                else:
                    msg_content = {
                        "role": getattr(msg, "role", "assistant"),
                        "content": getattr(msg, "content", "")
                    }
                choice_dict["message"] = msg_content
                
                choices_list.append(choice_dict)
        
        raw_response["choices"] = choices_list
        
        # Add usage if available
        if hasattr(response, "usage"):
            usage = response.usage
            if isinstance(usage, dict):
                usage_info: UsageInfo = {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                }
                raw_response["usage"] = usage_info
            else:
                usage_info = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None)
                }
                raw_response["usage"] = usage_info
        
        # Add ecological impact if available
        if hasattr(response, 'impacts'):
            raw_response["impact"] = self._extract_impacts(response)
        
        return content, raw_response
    
    def stream_with_raw(self, messages: List[ChatMessage], model: str) -> Generator[Tuple[str, RawStreamChunk], None, None]:
        """Streaming chat completion with raw chunks"""
        formatted = self.format_messages(messages)
        stream = self.client.chat.stream(
            model=model,
            messages=formatted  # type: ignore[arg-type]
        )
        
        for event in stream:
            content = ""
            raw_chunk: RawStreamChunk = {
                "object": "chat.completion.chunk",
                "model": model,
                "choices": []
            }
            
            if hasattr(event, "data") and event.data:
                # Add event metadata
                raw_chunk["id"] = getattr(event.data, "id", None)
                raw_chunk["created"] = getattr(event.data, "created", None)
                
                choices_list: List[StreamChoiceInfo] = []
                if hasattr(event.data, "choices") and event.data.choices:
                    for choice in event.data.choices:
                        delta = choice.delta if hasattr(choice, "delta") else choice
                        
                        choice_dict: StreamChoiceInfo = {
                            "index": getattr(choice, "index", 0),
                            "delta": {},
                            "finish_reason": getattr(choice, "finish_reason", None)
                        }
                        
                        # Extract delta content
                        if isinstance(delta, dict):
                            choice_dict["delta"] = {
                                "role": delta.get("role"),
                                "content": delta.get("content"),
                                "function_call": delta.get("function_call"),
                                "tool_calls": delta.get("tool_calls"),
                            }
                            content = delta.get("content", "")
                        else:
                            delta_content = getattr(delta, "content", None)
                            if delta_content:
                                content = delta_content
                            choice_dict["delta"] = {
                                "role": getattr(delta, "role", None),
                                "content": delta_content
                            }
                        
                        choices_list.append(choice_dict)
                
                raw_chunk["choices"] = choices_list
            
            if content:
                yield content, raw_chunk
