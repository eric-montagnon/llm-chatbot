from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator, List, Optional


@dataclass
class ChatMessage:
    """Represents a single chat message"""
    role: str
    content: str


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, api_key: str, default_model: Optional[str] = None):
        self.api_key = api_key
        self.default_model = default_model
        self._client: object = None
    
    @property
    def client(self) -> object:
        """Lazy initialization of client"""
        if self._client is None:
            self._client = self._initialize_client()
        return self._client
    
    @abstractmethod
    def _initialize_client(self) -> object:
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
    
    def format_messages(self, messages: List[ChatMessage]) -> list[dict[str, str]]:
        """Default message formatting - can be overridden"""
        return [{"role": m.role, "content": m.content} for m in messages]