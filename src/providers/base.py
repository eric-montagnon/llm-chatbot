from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (Dict, Generator, Generic, List, Optional, Sequence, Tuple,
                    TypedDict, TypeVar, Union)


@dataclass
class ChatMessage:
    """Represents a single chat message"""
    role: str
    content: str


# TypedDicts for raw API responses
class FunctionCall(TypedDict, total=False):
    """Function call information"""
    name: str
    arguments: str


class ToolCall(TypedDict, total=False):
    """Tool call information"""
    id: str
    type: str
    function: FunctionCall


class LogProbInfo(TypedDict, total=False):
    """Log probability information - can be a dict from serialization"""
    token: str
    logprob: float
    bytes: Optional[List[int]]
    top_logprobs: List['LogProbInfo']


# Type alias for serialized complex objects from API SDKs
SerializedDict = Dict[str, object]


class UsageInfo(TypedDict, total=False):
    """Token usage information"""
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]


class MessageContent(TypedDict, total=False):
    """Message content structure"""
    role: Optional[str]
    content: Optional[str]
    function_call: Union[FunctionCall, SerializedDict, None]
    tool_calls: Union[List[ToolCall], List[SerializedDict], None]


class ChoiceInfo(TypedDict, total=False):
    """Choice information in response"""
    index: int
    message: MessageContent
    finish_reason: Optional[str]
    logprobs: Union[LogProbInfo, SerializedDict, None]


class RawResponse(TypedDict, total=False):
    """Raw API response structure"""
    id: Optional[str]
    object: str
    created: Optional[int]
    model: str
    choices: Sequence[ChoiceInfo]
    usage: Optional[UsageInfo]
    system_fingerprint: Optional[str]


class DeltaContent(TypedDict, total=False):
    """Delta content for streaming"""
    role: Optional[str]
    content: Optional[str]
    function_call: Union[FunctionCall, SerializedDict, None]
    tool_calls: Union[List[ToolCall], List[SerializedDict], None]


class StreamChoiceInfo(TypedDict, total=False):
    """Choice information in streaming response"""
    index: int
    delta: DeltaContent
    finish_reason: Optional[str]
    logprobs: Union[LogProbInfo, SerializedDict, None]


class RawStreamChunk(TypedDict, total=False):
    """Raw streaming chunk structure"""
    id: Optional[str]
    object: str
    created: Optional[int]
    model: str
    choices: Sequence[StreamChoiceInfo]
    system_fingerprint: Optional[str]


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