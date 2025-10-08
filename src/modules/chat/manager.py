import time
from typing import Dict, Generator, List, Literal, Tuple, Union, overload

from modules.chat.types import (ChatMessage, CompleteInteraction, Interaction,
                                RequestData, StreamingInteraction)
from modules.config.settings import Config
from modules.config.types import LLMProviderInstance
from modules.providers.types import RawResponse, RawStreamChunk
from modules.tools.builtin import register_builtin_tools
from modules.tools.registry import ToolRegistry


class ChatManager:
    """Manages chat state and interactions with LLM providers"""
    
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        self.raw_interactions: List[Interaction] = []  # Store raw requests and responses
        self._provider_cache: Dict[str, LLMProviderInstance] = {}
        
        # Initialize tool registry
        self.tool_registry = ToolRegistry()
        register_builtin_tools(self.tool_registry)
    
    def get_provider(self, provider_name: str) -> LLMProviderInstance:
        if provider_name not in self._provider_cache:
            provider = Config.get_provider(provider_name)
            # Set tool registry on provider
            provider.set_tool_registry(self.tool_registry)
            self._provider_cache[provider_name] = provider
        return self._provider_cache[provider_name]
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
    
    def clear_chat(self):
        # Keep system prompt if it exists
        system_msg = None
        if self.messages and self.messages[0]["role"] == "system":
            system_msg = self.messages[0]
        
        self.messages.clear()
        self.raw_interactions.clear()  # Clear raw interactions too
        
        if system_msg:
            self.messages.append(system_msg)
    
    def update_system_prompt(self, prompt: str):
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = prompt
        else:
            self.messages.insert(0, {"role": "system", "content": prompt})
    
    def get_display_messages(self) -> List[Dict[str, str]]:
        return [msg for msg in self.messages if msg["role"] != "system"]
    
    def add_raw_interaction(self, interaction: Interaction):
        """Add a raw interaction (request/response pair) to history"""
        self.raw_interactions.append(interaction)
    
    def get_raw_interactions(self) -> List[Interaction]:
        """Get all raw interactions"""
        return self.raw_interactions
    
    @overload
    def generate_response_with_raw(
        self, 
        provider_name: str, 
        model: str, 
        stream: Literal[True]
    ) -> Generator[Tuple[str, RawStreamChunk], None, None]: ...
    
    @overload
    def generate_response_with_raw(
        self, 
        provider_name: str, 
        model: str, 
        stream: Literal[False]
    ) -> Tuple[str, RawResponse]: ...
    
    def generate_response_with_raw(
        self, 
        provider_name: str, 
        model: str, 
        stream: bool = True
    ) -> Union[Tuple[str, RawResponse], Generator[Tuple[str, RawStreamChunk], None, None]]:
        """Generate response with raw data and timing information"""
        provider = self.get_provider(provider_name)
        chat_messages = [
            ChatMessage(
                role=msg["role"],
                content=msg["content"]
            ) for msg in self.messages
        ]
        
        # Create request data
        request_data: RequestData = {
            "provider": provider_name,
            "model": model,
            "messages": self.messages,
            "stream": stream,
            "timestamp": time.time()
        }
        
        if stream:
            def stream_with_tracking() -> Generator[Tuple[str, RawStreamChunk], None, None]:
                start_time = time.time()
                chunks: List[RawStreamChunk] = []
                total_content = ""
                
                for content, raw_chunk in provider.stream_with_raw(chat_messages, model):
                    chunks.append(raw_chunk)
                    total_content += content
                    yield content, raw_chunk
                
                # After streaming completes, save the interaction
                end_time = time.time()
                interaction: StreamingInteraction = {
                    "type": "streaming",
                    "request": request_data,
                    "response": {
                        "chunks": chunks,
                        "total_chunks": len(chunks),
                        "final_content": total_content,
                        "duration_seconds": end_time - start_time
                    },
                    "timestamp": end_time
                }
                self.add_raw_interaction(interaction)
            
            return stream_with_tracking()
        else:
            start_time = time.time()
            content, raw_response = provider.complete_with_raw(chat_messages, model)
            end_time = time.time()
            
            # Save the interaction
            interaction: CompleteInteraction = {
                "type": "complete",
                "request": request_data,
                "response": {
                    "raw": raw_response,
                    "content": content,
                    "duration_seconds": end_time - start_time
                },
                "timestamp": end_time
            }
            self.add_raw_interaction(interaction)
            
            return content, raw_response
