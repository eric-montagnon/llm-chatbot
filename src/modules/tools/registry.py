from typing import Dict, List, Optional

from modules.tools.types import ToolDefinition


class ToolRegistry:
    """Registry for available tools"""
    
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
    
    def register(self, tool: ToolDefinition):
        """Register a tool"""
        self._tools[tool["name"]] = tool
    
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name"""
        return self._tools.get(name)
    
    def get_all_tools(self) -> List[ToolDefinition]:
        """Get all registered tools"""
        return list(self._tools.values())
    
    def execute_tool(self, name: str, arguments: Dict[str, object]) -> object:
        """Execute a tool with given arguments"""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")
        
        return tool["function"](**arguments)
    
    def to_openai_format(self) -> List[Dict[str, object]]:
        """Convert tools to OpenAI function calling format"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": {
                        "type": "object",
                        "properties": tool["parameters"],
                        "required": tool["required"]
                    }
                }
            }
            for tool in self._tools.values()
        ]
    
    def to_mistral_format(self) -> List[Dict[str, object]]:
        """Convert tools to Mistral function calling format"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": {
                        "type": "object",
                        "properties": tool["parameters"],
                        "required": tool["required"]
                    }
                }
            }
            for tool in self._tools.values()
        ]
