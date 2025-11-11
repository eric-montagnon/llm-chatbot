from typing import Any, Iterator, List, Optional

from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    print(f"Getting weather for {city}...")
    return f"It's always sunny in {city}!"


class LangChainProvider:
    """Provider class for LangChain agent interactions."""
    
    def __init__(self, model: str = "gpt-4", system_prompt: str = "You are a helpful assistant"):
        """
        Initialize the LangChain provider.
        
        Args:
            model: The model to use (default: gpt-4)
            system_prompt: System prompt for the agent
        """
        self.model = model
        self.system_prompt = system_prompt
        self.checkpointer = MemorySaver()
        self.agent = None
        self.current_thread_id = None
        self._messages: List[HumanMessage | AIMessage | SystemMessage] = []  # Accumulated messages from streaming

    def get_response_stream(
        self,
        user_query: str,
        thread_id: str = "default",
        update_messages: bool = True
    ) -> Iterator[Any]:
        """
        Get a streaming response from the LLM.
        
        Args:
            user_query: The user's question
            chat_history: Previous conversation messages
            thread_id: Thread ID for conversation persistence
            update_messages: If True, update self._messages with the stream
            
        Returns:
            Iterator yielding message chunks
        """
        # Create or recreate the agent
        self.agent = create_agent(
            model=self.model,
            tools=[get_weather],
            system_prompt=self.system_prompt,
            checkpointer=self.checkpointer,
        )
        self.current_thread_id = thread_id
        
        # Prepare messages
        self._messages.append(HumanMessage(content=user_query))
        
        # Stream the response
        stream = self.agent.stream(
            {"messages": [{"role": "user", "content": user_query}]},
            config={"configurable": {"thread_id": thread_id}},
            stream_mode="messages",
        )
        
        # If update_messages is True, wrap the stream to accumulate messages
        if update_messages:
            return self._stream_with_accumulation(stream)
        
        return stream
    
    def _stream_with_accumulation(self, stream: Iterator[Any]) -> Iterator[Any]:
        """
        Wrapper that accumulates messages while streaming.
        
        Args:
            stream: Original stream iterator
            
        Yields:
            Message chunks from the stream
        """
        
        for message, metadata in stream:
            yield message, metadata

            node = None
            if isinstance(metadata, dict):
                node = metadata.get('langgraph_node', 'unknown')

            if hasattr(message, 'content_blocks') and node == "model":
                content_blocks = getattr(message, 'content_blocks')

                if len(content_blocks) == 0:
                    continue

                block = content_blocks[0]

                if block.get('type') == "tool_call_chunk" : 
                    id = block.get('id')
                    name = block.get('name')
                    args = block.get('args', '')
                    previous_message = self._messages[-1] if self._messages else None
                    if previous_message is None or not isinstance(previous_message, AIMessage) or not (previous_message.tool_calls[0]):
                        self._messages.append(AIMessage(content= "", tool_calls= [{
                            "id": id,
                            "name": name,
                            "args": {"arguments": args, "response": ""}
                        }]))
                    else : 
                        previous_message.tool_calls[0]["args"]["arguments"] += args

                if block.get('type') == "text" : 
                    text = block.get('text', '')
                    previous_message = self._messages[-1] if self._messages else None
                    if previous_message is None or not isinstance(previous_message, AIMessage) or previous_message.tool_calls:
                        self._messages.append(AIMessage(content=text))
                    else:
                        previous_message.content += text
                print("Updated messages:", self._messages)

            if node == "tools":
                previous_message = self._messages[-1] if self._messages else None
                if not hasattr(message, 'content') or previous_message is None or not isinstance(previous_message, AIMessage) or not (previous_message.tool_calls[0]):
                    continue
                previous_message.tool_calls[0]["args"]["response"] += message.content

        # Update messages from state ONCE after streaming completes
        if self.agent and self.current_thread_id:
            state = self.agent.get_state({"configurable": {"thread_id": self.current_thread_id}})
            self._messages = state.values.get("messages", [])
            print("Final state after streaming:", self._messages)

    def get_messages(self) -> List[HumanMessage | AIMessage | SystemMessage]:
        return self._messages
    
    def clear_history(self, thread_id: Optional[str] = None) -> None:
        """
        Clear conversation history for a thread.
        
        Args:
            thread_id: Thread ID to clear (uses current if None)
        """
        # Note: MemorySaver doesn't have a built-in clear method
        # You might need to reinitialize the checkpointer or implement custom logic
        self.__init__(model=self.model, system_prompt=self.system_prompt)
    
    def set_model(self, model: str) -> None:
        """Update the model being used."""
        self.model = model
        
    def set_system_prompt(self, system_prompt: str) -> None:
        """Update the system prompt."""
        self.system_prompt = system_prompt