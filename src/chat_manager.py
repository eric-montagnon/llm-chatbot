from typing import Dict, Generator, List, Literal, Union, overload

from config import Config
from providers import MistralProvider, OpenAIProvider
from providers.base import ChatMessage


class ChatManager:
    """Manages chat state and interactions with LLM providers"""
    
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        self._provider_cache: Dict[str, Union[OpenAIProvider, MistralProvider]] = {}
    
    def get_provider(self, provider_name: str) -> Union[OpenAIProvider, MistralProvider]:
        if provider_name not in self._provider_cache:
            self._provider_cache[provider_name] = Config.get_provider(provider_name)
        return self._provider_cache[provider_name]
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
    
    def clear_chat(self):
        # Keep system prompt if it exists
        system_msg = None
        if self.messages and self.messages[0]["role"] == "system":
            system_msg = self.messages[0]
        
        self.messages.clear()
        
        if system_msg:
            self.messages.append(system_msg)
    
    def update_system_prompt(self, prompt: str):
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = prompt
        else:
            self.messages.insert(0, {"role": "system", "content": prompt})
    
    def get_display_messages(self) -> List[Dict[str, str]]:
        return [msg for msg in self.messages if msg["role"] != "system"]
    
    @overload
    def generate_response(
        self, 
        provider_name: str, 
        model: str, 
        stream: Literal[True]
    ) -> Generator[str, None, None]: ...
    
    @overload
    def generate_response(
        self, 
        provider_name: str, 
        model: str, 
        stream: Literal[False]
    ) -> str: ...
    
    def generate_response(
        self, 
        provider_name: str, 
        model: str, 
        stream: bool = True
    ) -> Union[str, Generator[str, None, None]]:
        provider = self.get_provider(provider_name)
        chat_messages = [ChatMessage(**msg) for msg in self.messages]
        
        if stream:
            return provider.stream(chat_messages, model)
        else:
            return provider.complete(chat_messages, model)