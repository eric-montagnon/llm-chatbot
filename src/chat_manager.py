import time
from typing import (Dict, Generator, List, Literal, Tuple, TypedDict, Union,
                    overload)

from config import Config
from providers import MistralProvider, OpenAIProvider
from providers.base import ChatMessage, RawResponse, RawStreamChunk


class RequestData(TypedDict):
    """Structure for request data"""
    provider: str
    model: str
    messages: List[Dict[str, str]]
    stream: bool
    timestamp: float


class StreamingResponse(TypedDict):
    """Structure for streaming response data"""
    chunks: List[RawStreamChunk]
    total_chunks: int
    final_content: str
    duration_seconds: float


class CompleteResponse(TypedDict):
    """Structure for complete response data"""
    raw: RawResponse
    content: str
    duration_seconds: float


class StreamingInteraction(TypedDict):
    """Structure for streaming interaction"""
    type: Literal["streaming"]
    request: RequestData
    response: StreamingResponse
    timestamp: float


class CompleteInteraction(TypedDict):
    """Structure for complete interaction"""
    type: Literal["complete"]
    request: RequestData
    response: CompleteResponse
    timestamp: float


Interaction = Union[StreamingInteraction, CompleteInteraction]


class ChatManager:
    """Manages chat state and interactions with LLM providers"""
    
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        self.raw_interactions: List[Interaction] = []  # Store raw requests and responses
        self._provider_cache: Dict[str, Union[OpenAIProvider, MistralProvider]] = {}
    
    def get_provider(self, provider_name: str) -> Union[OpenAIProvider, MistralProvider]:
        if provider_name not in self._provider_cache:
            self._provider_cache[provider_name] = Config.get_provider(provider_name)
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
    
    @overload
    def generate_response(
        self, 
        provider_name: str, 
        model: str, 
        stream: Literal[True]
    ) -> Generator[str, None, None]: ...
    
    @overload
    def generate_response(
        self, 
        provider_name: str, 
        model: str, 
        stream: Literal[False]
    ) -> str: ...
    
    def generate_response(
        self, 
        provider_name: str, 
        model: str, 
        stream: bool = True
    ) -> Union[str, Generator[str, None, None]]:
        provider = self.get_provider(provider_name)
        chat_messages = [ChatMessage(**msg) for msg in self.messages]
        
        if stream:
            return provider.stream(chat_messages, model)
        else:
            return provider.complete(chat_messages, model)
    
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
        chat_messages = [ChatMessage(**msg) for msg in self.messages]
        
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
