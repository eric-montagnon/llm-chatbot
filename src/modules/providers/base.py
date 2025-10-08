from abc import ABC, abstractmethod
from typing import Generator, Generic, List, Optional, Tuple, TypeVar

from .types import ChatMessage, RawResponse, RawStreamChunk

# Generic type variable for the client
ClientType = TypeVar('ClientType')


class LLMProvider(ABC, Generic[ClientType]):
    """Abstract base class for LLM providers"""
    
    def __init__(self, api_key: str, default_model: Optional[str] = None):
        self.api_key = api_key
        self.default_model = default_model
        self._client: Optional[ClientType] = None
    
    @property
    def client(self) -> ClientType:
        """Lazy initialization of client"""
        if self._client is None:
            self._client = self._initialize_client()
        return self._client
    
    @abstractmethod
    def _initialize_client(self) -> ClientType:
        """Initialize the provider-specific client"""
        pass
    
    @abstractmethod
    def complete(self, messages: List[ChatMessage], model: str) -> str:
        """Non-streaming completion"""
        pass
    
    @abstractmethod
    def stream(self, messages: List[ChatMessage], model: str) -> Generator[str, None, None]:
        """Streaming completion"""
        pass
    
    @abstractmethod
    def complete_with_raw(self, messages: List[ChatMessage], model: str) -> Tuple[str, RawResponse]:
        """Non-streaming completion with raw response"""
        pass
    
    @abstractmethod
    def stream_with_raw(self, messages: List[ChatMessage], model: str) -> Generator[Tuple[str, RawStreamChunk], None, None]:
        """Streaming completion with raw response chunks"""
        pass
    
    def format_messages(self, messages: List[ChatMessage]) -> list[dict[str, str]]:
        """Default message formatting - can be overridden"""
        return [{"role": m.role, "content": m.content} for m in messages]