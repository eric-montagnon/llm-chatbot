from typing import Dict, Generator, List, Sequence, Tuple

from ecologits import EcoLogits
from mistralai import Mistral

from .base import LLMProvider
from .types import (ChatMessage, ChoiceInfo, EcologicalImpact, MessageContent,
                    RawResponse, RawStreamChunk, StreamChoiceInfo,
                    ToolResultInfo, UsageInfo)

DEFAULT_TEMPERATURE: float = 0

class MistralProvider(LLMProvider[Mistral]):
    
    def _initialize_client(self) -> Mistral:
        if not self.api_key:
            raise RuntimeError("Missing MISTRAL_API_KEY")
        
        # Initialize EcoLogits tracer
        EcoLogits.init()
        
        return Mistral(api_key=self.api_key)
    
    def _format_for_mistral(self, messages: List[ChatMessage]) -> Sequence[Dict[str, object]]:
        """Convert ChatMessage to Mistral's expected message format"""
        result: List[Dict[str, object]] = []
        
        for msg in messages:
            msg_dict: Dict[str, object] = {
                "role": msg.role,
            }
            # Only add content if it's not empty - particularly important for assistant messages with tool calls
            if msg.content:
                msg_dict["content"] = msg.content
            
            if msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
                # Tool messages always need content
                if not msg.content:
                    msg_dict["content"] = ""
            if msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls
            result.append(msg_dict)
        
        return result
    
    def _extract_impacts(self, response: object) -> EcologicalImpact:
        """Extract ecological impacts from response"""
        if hasattr(response, 'impacts'):
            impacts_attr = getattr(response, 'impacts', None)
            if impacts_attr is not None:
                # Helper function to extract float from impact value
                def extract_value(impact_obj: object) -> float:
                    """Extract float value from impact object, handling RangeValue"""
                    if impact_obj is None:
                        return 0.0
                    value = getattr(impact_obj, 'value', 0.0)
                    # Value can be int, float, or RangeValue
                    if isinstance(value, (int, float)):
                        return float(value)
                    # RangeValue has .mean property
                    if hasattr(value, 'mean'):
                        return float(getattr(value, 'mean', 0.0))
                    return 0.0
                
                return {
                    "energy_kwh": extract_value(getattr(impacts_attr, 'energy', None)),
                    "gwp_kgco2eq": extract_value(getattr(impacts_attr, 'gwp', None)),
                    "adpe_kgsbeq": extract_value(getattr(impacts_attr, 'adpe', None)),
                    "pe_mj": extract_value(getattr(impacts_attr, 'pe', None)),
                }
        # Return default values if no impacts available
        return {
            "energy_kwh": 0.0,
            "gwp_kgco2eq": 0.0,
            "adpe_kgsbeq": 0.0,
            "pe_mj": 0.0,
        }
    
    def complete_with_raw(self, messages: List[ChatMessage], model: str) -> Tuple[str, RawResponse]:
        """Non-streaming chat completion with raw response"""
        formatted = self._format_for_mistral(messages)
        
        # Prepare request kwargs
        kwargs: Dict[str, object] = {
            "model": model,
            "messages": formatted,
            "temperature": DEFAULT_TEMPERATURE
        }
        
        # Add tools if registry is available
        if self.tool_registry:
            kwargs["tools"] = self.tool_registry.to_mistral_format()
            kwargs["tool_choice"] = "auto"
        
        response = self.client.chat.complete(**kwargs)  # type: ignore[arg-type]
        
        # Check for tool calls
        message = response.choices[0].message
        if hasattr(message, 'tool_calls') and message.tool_calls:
            return self._handle_tool_calls(messages, model, response)
        
        # Extract content
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
    
    def _handle_tool_calls(
        self, 
        messages: List[ChatMessage], 
        model: str, 
        response: object
    ) -> Tuple[str, RawResponse]:
        """Handle tool calls and execute them"""
        import json
        
        if not self.tool_registry:
            raise RuntimeError("Tool registry not set")
        
        # Extract message and tool calls - defensive access with getattr
        choices = getattr(response, 'choices', None)
        if not choices:
            raise RuntimeError("Response has no choices")
        
        message = getattr(choices[0], 'message', None)
        if message is None:
            raise RuntimeError("Choice has no message")
            
        if not hasattr(message, 'tool_calls') or not message.tool_calls:
            raise RuntimeError("Tool calls expected but not found")
        
        # Execute each tool call
        tool_messages: List[ChatMessage] = []
        tool_results: List[ToolResultInfo] = []
        
        for tool_call in message.tool_calls:
            # Only handle function tool calls
            if not hasattr(tool_call, 'function'):
                continue
            
            function_obj = getattr(tool_call, 'function', None)
            if function_obj is None:
                continue
                
            function_name = str(getattr(function_obj, 'name', ''))
            # Mistral arguments might be a dict or string
            function_args_raw = getattr(function_obj, 'arguments', '{}')
            if isinstance(function_args_raw, str):
                function_args = json.loads(function_args_raw)
            else:
                function_args = function_args_raw
            
            # Execute the tool
            result = self.tool_registry.execute_tool(function_name, function_args)
            
            # Store result for display
            tool_results.append({
                "name": function_name,
                "arguments": function_args,
                "result": str(result)
            })
            
            # Add tool result message
            tool_call_id = getattr(tool_call, 'id', '')
            tool_messages.append(ChatMessage(
                role="tool",
                content=str(result),
                tool_call_id=tool_call_id
            ))
        
        # Make another call with tool results
        tool_call_list: List[Dict[str, object]] = [
            {
                "id": getattr(tool_call, 'id', ''),
                "type": "function",
                "function": {
                    "name": str(getattr(getattr(tool_call, 'function', None), 'name', '')),
                    "arguments": getattr(getattr(tool_call, 'function', None), 'arguments', '{}')
                }
            } for tool_call in message.tool_calls
            if hasattr(tool_call, 'function')
        ]
        
        new_messages = messages + [
            ChatMessage(
                role="assistant",
                content=getattr(message, "content", "") or "",
                tool_calls=tool_call_list
            )
        ] + tool_messages
        
        formatted = self._format_for_mistral(new_messages)
        
        final_response = self.client.chat.complete(
            model=model,
            messages=formatted,  # type: ignore[arg-type]
            temperature=DEFAULT_TEMPERATURE
        )
        
        # Extract content from final response
        final_message = final_response.choices[0].message
        if isinstance(final_message, dict):
            content = final_message.get("content", "")
            content = content if isinstance(content, str) else ""
        else:
            content = getattr(final_message, "content", None)
            content = content if isinstance(content, str) else ""
        
        # Build raw response
        raw_response: RawResponse = {
            "id": getattr(final_response, "id", None),
            "object": getattr(final_response, "object", "chat.completion"),
            "created": getattr(final_response, "created", None),
            "model": getattr(final_response, "model", model),
            "choices": [],
        }
        
        # Add choices
        choices_list: List[ChoiceInfo] = []
        if hasattr(final_response, "choices") and final_response.choices:
            for choice in final_response.choices:
                msg = choice.message
                msg_content: MessageContent = {}
                
                if isinstance(msg, dict):
                    msg_content["role"] = msg.get("role", "assistant")
                    msg_content["content"] = msg.get("content")
                else:
                    msg_content["role"] = getattr(msg, "role", "assistant")
                    msg_content["content"] = getattr(msg, "content", None)
                
                choice_info: ChoiceInfo = {
                    "index": getattr(choice, "index", 0),
                    "message": msg_content,
                    "finish_reason": getattr(choice, "finish_reason", None),
                }
                choices_list.append(choice_info)
        
        raw_response["choices"] = choices_list
        
        # Add usage if available
        if hasattr(final_response, "usage"):
            usage = final_response.usage
            if isinstance(usage, dict):
                usage_info: UsageInfo = {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens")
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
        if hasattr(final_response, 'impacts'):
            raw_response["impact"] = self._extract_impacts(final_response)
        
        # Add tool execution info
        raw_response["tool_results"] = tool_results
        
        return content, raw_response
    
    def stream_with_raw(self, messages: List[ChatMessage], model: str) -> Generator[Tuple[str, RawStreamChunk], None, None]:
        """Streaming chat completion with raw chunks"""
        formatted = self._format_for_mistral(messages)
        
        # Prepare request kwargs
        kwargs: Dict[str, object] = {
            "model": model,
            "messages": formatted,
            "temperature": DEFAULT_TEMPERATURE
        }
        
        # Add tools if registry is available
        if self.tool_registry:
            kwargs["tools"] = self.tool_registry.to_mistral_format()
            kwargs["tool_choice"] = "auto"
        
        stream = self.client.chat.stream(**kwargs)  # type: ignore[arg-type]
        
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
        
        # Note: Mistral streaming with tool calls may require different handling
        # Tool calls in streaming mode are passed through in the chunks but not automatically executed
        # This is a known limitation that would require accumulating and processing tool call deltas