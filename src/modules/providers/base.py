from abc import ABC, abstractmethod
from typing import (TYPE_CHECKING, Dict, Generator, Generic, List, Optional,
                    Sequence, Tuple, TypeVar)

from .types import ChatMessage, RawResponse, RawStreamChunk

if TYPE_CHECKING:
    from modules.tools.registry import ToolRegistry

# Generic type variable for the client
ClientType = TypeVar('ClientType')


class LLMProvider(ABC, Generic[ClientType]):
    """Abstract base class for LLM providers"""
    
    def __init__(self, api_key: str, default_model: Optional[str] = None):
        self.api_key = api_key
        self.default_model = default_model
        self._client: Optional[ClientType] = None
        self.tool_registry: Optional["ToolRegistry"] = None
    
    def set_tool_registry(self, registry: "ToolRegistry"):
        """Set the tool registry for this provider"""
        self.tool_registry = registry
    
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
    def complete_with_raw(self, messages: List[ChatMessage], model: str) -> Tuple[str, RawResponse]:
        """Non-streaming completion with raw response"""
        pass
    
    @abstractmethod
    def stream_with_raw(self, messages: List[ChatMessage], model: str) -> Generator[Tuple[str, RawStreamChunk], None, None]:
        """Streaming completion with raw response chunks"""
        pass
    
    def format_messages(self, messages: List[ChatMessage]) -> Sequence[Dict[str, object]]:
        """Default message formatting - can be overridden"""
        formatted: List[Dict[str, object]] = []
        for m in messages:
            msg_dict: Dict[str, object] = {"role": m.role, "content": m.content}
            if m.tool_call_id:
                msg_dict["tool_call_id"] = m.tool_call_id
            if m.tool_calls:
                msg_dict["tool_calls"] = m.tool_calls
            formatted.append(msg_dict)
        return formatted