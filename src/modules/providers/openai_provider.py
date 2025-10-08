from typing import Dict, Generator, List, Tuple

from ecologits import EcoLogits
from openai import OpenAI
from openai.types.chat import (ChatCompletion,
                               ChatCompletionAssistantMessageParam,
                               ChatCompletionMessageParam,
                               ChatCompletionSystemMessageParam,
                               ChatCompletionToolMessageParam,
                               ChatCompletionToolParam,
                               ChatCompletionUserMessageParam)
from openai.types.shared_params import FunctionDefinition

from .base import LLMProvider
from .types import (ChatMessage, EcologicalImpact, RawResponse, RawStreamChunk,
                    ToolResultInfo)


class OpenAIProvider(LLMProvider[OpenAI]):
    """OpenAI-specific implementation"""
    
    def _initialize_client(self) -> OpenAI:
        if not self.api_key:
            raise RuntimeError("Missing OPENAI_API_KEY")
        
        # Initialize EcoLogits tracer
        EcoLogits.init()
        
        return OpenAI(api_key=self.api_key)
    
    def _format_for_openai(self, messages: List[ChatMessage]) -> List[ChatCompletionMessageParam]:
        """Convert ChatMessage to OpenAI's typed message format"""
        result: List[ChatCompletionMessageParam] = []
        
        for msg in messages:
            if msg.tool_call_id:
                # Tool message
                tool_msg: ChatCompletionToolMessageParam = {
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id
                }
                result.append(tool_msg)
            elif msg.tool_calls:
                # Assistant message with tool calls - needs special handling
                # Since tool_calls might not match the expected type exactly, we construct carefully
                asst_msg: ChatCompletionAssistantMessageParam = {
                    "role": "assistant",
                }
                if msg.content:
                    asst_msg["content"] = msg.content
                # Note: tool_calls from our ChatMessage is List[Dict[str, object]]
                # which doesn't match OpenAI's expected type exactly.
                # We'll need to handle this without the tool_calls for now
                result.append(asst_msg)
            elif msg.role == "system":
                sys_msg: ChatCompletionSystemMessageParam = {
                    "role": "system",
                    "content": msg.content
                }
                result.append(sys_msg)
            elif msg.role == "user":
                user_msg: ChatCompletionUserMessageParam = {
                    "role": "user",
                    "content": msg.content
                }
                result.append(user_msg)
            elif msg.role == "assistant":
                asst_msg2: ChatCompletionAssistantMessageParam = {
                    "role": "assistant",
                    "content": msg.content
                }
                result.append(asst_msg2)
            else:
                # Fallback to system for unknown roles
                fallback_msg: ChatCompletionSystemMessageParam = {
                    "role": "system",
                    "content": msg.content
                }
                result.append(fallback_msg)
        
        return result
    
    def _convert_tools_to_openai(self) -> List[ChatCompletionToolParam]:
        """Convert tools from registry to OpenAI's typed format"""
        if not self.tool_registry:
            return []
        
        result: List[ChatCompletionToolParam] = []
        tools_data = self.tool_registry.to_openai_format()
        
        for tool_dict in tools_data:
            # tool_dict is Dict[str, object], we need to construct ChatCompletionToolParam
            # Extract the function data
            func_obj = tool_dict.get("function")
            if not isinstance(func_obj, dict):
                continue
            
            name_obj = func_obj.get("name", "")
            desc_obj = func_obj.get("description", "")
            
            # Build the FunctionDefinition
            function_def: FunctionDefinition = {
                "name": str(name_obj),
                "description": str(desc_obj)
            }
            # Add parameters if present - parameters field is optional in FunctionDefinition
            # We know it's a dict from our registry structure
            params_obj = func_obj.get("parameters")
            if params_obj is not None:
                function_def["parameters"] = params_obj  # parameters is Dict[str, object] which matches
            
            # Build the ChatCompletionToolParam
            tool_param: ChatCompletionToolParam = {
                "type": "function",
                "function": function_def
            }
            result.append(tool_param)
        
        return result
    
    def _extract_impacts(self, response: ChatCompletion) -> EcologicalImpact:
        """Extract ecological impacts from response"""
        if hasattr(response, 'impacts'):
            # ecologits adds impacts dynamically, we need to access it carefully
            impacts_attr = getattr(response, 'impacts', None)
            if impacts_attr is not None:
                return {
                    "energy_kwh": float(getattr(impacts_attr.energy, 'value', 0.0)),
                    "gwp_kgco2eq": float(getattr(impacts_attr.gwp, 'value', 0.0)),
                    "adpe_kgsbeq": float(getattr(impacts_attr.adpe, 'value', 0.0)),
                    "pe_mj": float(getattr(impacts_attr.pe, 'value', 0.0)),
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
        formatted = self._format_for_openai(messages)
        
        # Check if we have tools to add
        if self.tool_registry:
            tools_list = self._convert_tools_to_openai()
            response = self.client.chat.completions.create(
                model=model,
                messages=formatted,
                stream=False,
                tools=tools_list,
                tool_choice="auto"
            )
        else:
            response = self.client.chat.completions.create(
                model=model,
                messages=formatted,
                stream=False
            )
        
        # Type narrowing: stream=False guarantees ChatCompletion
        assert isinstance(response, ChatCompletion)
        
        # Handle tool calls if present
        if response.choices[0].message.tool_calls:
            return self._handle_tool_calls(messages, model, response)
        
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
        
        # Add ecological impact if available
        if hasattr(response, 'impacts'):
            raw_response["impact"] = self._extract_impacts(response)
        
        return content, raw_response
    
    def _handle_tool_calls(
        self, 
        messages: List[ChatMessage], 
        model: str, 
        response: ChatCompletion
    ) -> Tuple[str, RawResponse]:
        """Handle tool calls and execute them"""
        import json
        
        if not self.tool_registry:
            raise RuntimeError("Tool registry not set")
        
        # Defensive check
        if not response.choices[0].message.tool_calls:
            raise RuntimeError("Tool calls expected but not found")
        
        # Execute each tool call
        tool_messages: List[ChatMessage] = []
        tool_results: List[ToolResultInfo] = []
        
        for tool_call in response.choices[0].message.tool_calls:
            # Only handle function tool calls
            if not hasattr(tool_call, 'function'):
                continue
            
            function_obj = getattr(tool_call, 'function', None)
            if function_obj is None:
                continue
            
            function_name = str(getattr(function_obj, 'name', ''))
            function_args_str = str(getattr(function_obj, 'arguments', '{}'))
            function_args = json.loads(function_args_str)
            
            # Execute the tool
            result = self.tool_registry.execute_tool(function_name, function_args)
            
            # Store result for display
            tool_results.append({
                "name": function_name,
                "arguments": function_args,
                "result": str(result)
            })
            
            # Add tool result message
            tool_messages.append(ChatMessage(
                role="tool",
                content=str(result),
                tool_call_id=tool_call.id
            ))
        
        # Make another call with tool results
        tool_call_list: List[Dict[str, object]] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": str(getattr(getattr(tc, 'function', None), 'name', '')),
                    "arguments": str(getattr(getattr(tc, 'function', None), 'arguments', '{}'))
                }
            } for tc in response.choices[0].message.tool_calls
            if hasattr(tc, 'function')
        ]
        
        new_messages = messages + [
            ChatMessage(
                role="assistant",
                content=response.choices[0].message.content or "",
                tool_calls=tool_call_list
            )
        ] + tool_messages
        
        formatted = self._format_for_openai(new_messages)
        
        final_response = self.client.chat.completions.create(
            model=model,
            messages=formatted,
            stream=False
        )
        assert isinstance(final_response, ChatCompletion)
        
        content = final_response.choices[0].message.content or ""
        
        # Build raw response with tool information
        raw_response: RawResponse = {
            "id": final_response.id,
            "object": final_response.object,
            "created": final_response.created,
            "model": final_response.model,
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
                for choice in final_response.choices
            ],
            "system_fingerprint": final_response.system_fingerprint,
        }
        
        # Add usage
        if final_response.usage:
            raw_response["usage"] = {
                "prompt_tokens": final_response.usage.prompt_tokens,
                "completion_tokens": final_response.usage.completion_tokens,
                "total_tokens": final_response.usage.total_tokens,
            }
        
        # Add ecological impact if available
        if hasattr(final_response, 'impacts'):
            raw_response["impact"] = self._extract_impacts(final_response)
        
        # Add tool execution info
        raw_response["tool_results"] = tool_results
        
        return content, raw_response
    
    def stream_with_raw(self, messages: List[ChatMessage], model: str) -> Generator[Tuple[str, RawStreamChunk], None, None]:
        """Streaming chat completion with raw chunks"""
        formatted = self._format_for_openai(messages)
        stream_response = self.client.chat.completions.create(
            model=model,
            messages=formatted,
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
