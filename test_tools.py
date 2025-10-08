"""
Test script for tool calling functionality
"""

from modules.tools.builtin import register_builtin_tools
from modules.tools.registry import ToolRegistry


def test_tool_registry():
    """Test that tools are registered correctly"""
    print("Testing Tool Registry...")
    
    registry = ToolRegistry()
    register_builtin_tools(registry)
    
    # Check registered tools
    tools = registry.get_all_tools()
    print(f"\nRegistered {len(tools)} tools:")
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description']}")
    
    # Test tool execution
    print("\n\nTesting tool execution...")
    
    # Test weather tool
    result = registry.execute_tool("get_current_weather", {"location": "San Francisco, CA", "unit": "fahrenheit"})
    print(f"\nWeather tool result: {result}")
    
    # Test calculator
    result = registry.execute_tool("calculate", {"operation": "add", "a": 5, "b": 3})
    print(f"Calculator tool result: {result}")
    
    # Test time tool
    result = registry.execute_tool("get_current_time", {"timezone": "UTC"})
    print(f"Time tool result: {result}")
    
    # Test OpenAI format conversion
    print("\n\nOpenAI format:")
    openai_tools = registry.to_openai_format()
    print(f"Number of tools: {len(openai_tools)}")
    if openai_tools:
        first_tool = openai_tools[0]
        if isinstance(first_tool, dict) and isinstance(first_tool.get('function'), dict):
            function_dict = first_tool['function']
            if isinstance(function_dict, dict):
                print(f"First tool: {function_dict.get('name')}")
    
    # Test Mistral format conversion
    print("\nMistral format:")
    mistral_tools = registry.to_mistral_format()
    print(f"Number of tools: {len(mistral_tools)}")
    if mistral_tools:
        first_tool = mistral_tools[0]
        if isinstance(first_tool, dict) and isinstance(first_tool.get('function'), dict):
            function_dict = first_tool['function']
            if isinstance(function_dict, dict):
                print(f"First tool: {function_dict.get('name')}")
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_tool_registry()
