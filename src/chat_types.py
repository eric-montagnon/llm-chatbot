"""
Type definitions for chat messages and interactions.

This module contains types for managing chat state, messages, and interactions
between the user and LLM providers.
"""

from typing import Dict, List, Literal, TypedDict, Union

from providers.types import ChatMessage, RawResponse, RawStreamChunk

# Re-export ChatMessage for convenience
__all__ = [
    "ChatMessage",
    "RequestData",
    "StreamingResponse",
    "CompleteResponse",
    "StreamingInteraction",
    "CompleteInteraction",
    "Interaction",
]


# ============================================================================
# Request and Interaction Types
# ============================================================================

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


# Union type for all interaction types
Interaction = Union[StreamingInteraction, CompleteInteraction]
