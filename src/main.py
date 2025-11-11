from typing import List

import streamlit as st
from langchain.messages import (AIMessage, HumanMessage, SystemMessage,
                                ToolMessage)

from modules.config import Config
from modules.providers.langchain_class import LangChainProvider
from modules.ui import ChatUI, Sidebar


def show_message(message: HumanMessage | AIMessage | SystemMessage, messages: List[HumanMessage | AIMessage | SystemMessage]) -> None:
    if isinstance(message, HumanMessage):
        # Handle content that could be string or list
        content = message.content if isinstance(message.content, str) else str(message.content)
        ChatUI.display_message("user", content)
    elif isinstance(message, AIMessage):
        # Display tool calls if present
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                # Find the matching ToolMessage response by tool_call_id
                tool_response = ""
                for msg in messages:
                    if isinstance(msg, ToolMessage) and hasattr(msg, 'tool_call_id') and msg.tool_call_id == tool_call.get('id'):
                        tool_response = msg.content if isinstance(msg.content, str) else str(msg.content)
                        break
                
                ChatUI.display_tool_calls(tool_call, in_chat_context=False, response=tool_response)

        if message.content:
            content = message.content if isinstance(message.content, str) else str(message.content)
            ChatUI.display_ai_message_with_costs(message)

def show_messages_in_UI(messages: List[HumanMessage | AIMessage | SystemMessage]) -> None:
   for msg in messages:
        show_message(msg, messages)
            
st.set_page_config(
    page_title="LLM Chatbot with Raw View", 
    page_icon="💬", 
    layout="wide"  # Change to wide layout for two-column view
)
st.title("💬 LLM Chatbot with Request Viewer")
st.caption("See the exact messages being sent to the LLM")

if "langchain_provider" not in st.session_state:
    # Initialize LangChain provider
    st.session_state.langchain_provider = LangChainProvider()
    st.session_state.langchain_provider.set_system_prompt(Config.DEFAULT_SYSTEM_PROMPT)

provider, model, system_prompt, clear_pressed = Sidebar.render()

if clear_pressed:
    st.session_state.langchain_provider.clear_history()
    st.rerun()

st.session_state.langchain_provider.set_system_prompt(system_prompt)

# Create two columns for chat and raw view
chat_col, raw_col = st.columns([1, 1], gap="medium")

with chat_col:
    st.header("📨 Chat Interface")
    
    # Display existing messages
    display_messages: List[HumanMessage | AIMessage | SystemMessage] = st.session_state.langchain_provider.get_messages()
    
    # Display existing messages
    show_messages_in_UI(display_messages)
    
    # Chat input
    user_input = st.chat_input("Type your message…")
    
    if user_input:
        try:
            placeholder = st.empty()
            
            st.session_state.langchain_provider.set_model(model)
            
            response_stream = st.session_state.langchain_provider.get_response_stream(
                user_input, 
                thread_id="main_thread"
            )
            
            for message, metadata in response_stream:
                # Get all messages since the last HumanMessage
                all_messages = st.session_state.langchain_provider.get_messages()
                
                # Find the index of the last HumanMessage
                last_human_idx = -1
                for i in range(len(all_messages) - 1, -1, -1):
                    if isinstance(all_messages[i], HumanMessage):
                        last_human_idx = i
                        break
                
                # Get messages after the last HumanMessage
                messages_to_display = all_messages[last_human_idx + 1:] if last_human_idx != -1 else []
                
                # Display all messages since last HumanMessage
                with placeholder.container():
                    for msg in messages_to_display:
                        show_message(msg, all_messages)
            
            st.rerun()
            
        except Exception as e:
            ChatUI.display_error(e, show_details=True)