from typing import Callable, Dict, List, TypedDict


class ToolParameter(TypedDict, total=False):
    type: str
    description: str
    enum: List[str]  # Optional for string enums


class ToolDefinition(TypedDict):
    name: str
    description: str
    parameters: Dict[str, ToolParameter]
    required: List[str]
    function: Callable[..., object]


class ToolCall(TypedDict):
    id: str
    name: str
    arguments: Dict[str, object]


class ToolResult(TypedDict):
    tool_call_id: str
    content: str
