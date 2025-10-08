"""
Type definitions for LLM provider API responses.

This module contains all TypedDict definitions for raw API responses from LLM providers,
including streaming and non-streaming response structures.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, TypedDict, Union

# ============================================================================
# Message Types
# ============================================================================

@dataclass
class ChatMessage:
    """Represents a single chat message"""
    role: str
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, object]]] = None


# ============================================================================
# Basic API Types
# ============================================================================

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


class EcologicalImpact(TypedDict):
    """Ecological impact information from ecologits"""
    energy_kwh: float
    gwp_kgco2eq: float
    adpe_kgsbeq: float
    pe_mj: float


# ============================================================================
# Non-Streaming Response Types
# ============================================================================

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


class ToolResultInfo(TypedDict):
    """Tool execution result information"""
    name: str
    arguments: Dict[str, object]
    result: str


class RawResponse(TypedDict, total=False):
    """Raw API response structure"""
    id: Optional[str]
    object: str
    created: Optional[int]
    model: str
    choices: Sequence[ChoiceInfo]
    usage: Optional[UsageInfo]
    system_fingerprint: Optional[str]
    impact: Optional[EcologicalImpact]
    tool_results: Optional[List[ToolResultInfo]]


# ============================================================================
# Streaming Response Types
# ============================================================================

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
