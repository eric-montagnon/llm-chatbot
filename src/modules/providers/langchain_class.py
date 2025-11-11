from typing import Any, Iterator, List, Optional

from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from modules.providers.tools import calculate, get_current_time, get_weather


class LangChainProvider:
    """Provider class for LangChain agent interactions."""
    
    def __init__(self, model: str = "gpt-4", system_prompt: str = "You are a helpful assistant"):
        """
        Initialize the LangChain provider.
        
        Args:
            model: The model to use (default: gpt-4)
            system_prompt: System prompt for the agent
        """
        self.model_name = model
        self.system_prompt = system_prompt
        self.checkpointer = MemorySaver()
        self.agent = None
        self.current_thread_id = None
        self._messages: List[HumanMessage | AIMessage | SystemMessage] = []  # Accumulated messages from streaming

    def _get_model_instance(self):
        """Get the appropriate model instance based on model name."""
        # Determine if it's a Mistral or OpenAI model
        mistral_models = ["codestral-latest", "mistral-medium-latest", "mistral-small-latest", "mistral-large-latest"]
        
        if self.model_name in mistral_models or self.model_name.startswith("mistral-") or self.model_name.startswith("codestral-"):
            return ChatMistralAI(
                model_name=self.model_name
            )
        else:
            # Default to OpenAI
            return ChatOpenAI(
                model=self.model_name
            )

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
        # Create or recreate the agent with proper model instance
        model_instance = self._get_model_instance()
        
        self.agent = create_agent(
            model=model_instance,
            tools=[get_weather, get_current_time, calculate],
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
        self.__init__(model=self.model_name, system_prompt=self.system_prompt)
    
    def set_model(self, model: str) -> None:
        """Update the model being used."""
        self.model_name = model
        
    def set_system_prompt(self, system_prompt: str) -> None:
        """Update the system prompt."""
        self.system_prompt = system_prompt