import json
from datetime import datetime

from modules.tools.registry import ToolRegistry
from modules.tools.types import ToolDefinition


def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a location"""
    # Mock implementation
    return json.dumps({
        "location": location,
        "temperature": 22,
        "unit": unit,
        "condition": "sunny"
    })


def get_current_time(timezone: str = "UTC") -> str:
    """Get the current time in a specific timezone"""
    # Mock implementation - simplified version
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(operation: str, a: float, b: float) -> str:
    """Perform basic arithmetic operations"""
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else "Error: Division by zero"
    }
    
    if operation not in operations:
        return f"Error: Unknown operation '{operation}'"
    
    result = operations[operation](a, b)
    return str(result)


def register_builtin_tools(registry: ToolRegistry):
    """Register all built-in tools"""
    
    weather_tool: ToolDefinition = {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
                "type": "string",
                "description": "The temperature unit",
                "enum": ["celsius", "fahrenheit"]
            }
        },
        "required": ["location"],
        "function": get_current_weather
    }
    
    time_tool: ToolDefinition = {
        "name": "get_current_time",
        "description": "Get the current time in a specific timezone",
        "parameters": {
            "timezone": {
                "type": "string",
                "description": "The timezone, e.g. UTC, America/New_York"
            }
        },
        "required": [],
        "function": get_current_time
    }
    
    calculator_tool: ToolDefinition = {
        "name": "calculate",
        "description": "Perform basic arithmetic operations",
        "parameters": {
            "operation": {
                "type": "string",
                "description": "The operation to perform",
                "enum": ["add", "subtract", "multiply", "divide"]
            },
            "a": {
                "type": "number",
                "description": "First number"
            },
            "b": {
                "type": "number",
                "description": "Second number"
            }
        },
        "required": ["operation", "a", "b"],
        "function": calculate
    }
    
    registry.register(weather_tool)
    registry.register(time_tool)
    registry.register(calculator_tool)
