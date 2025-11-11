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

                ChatUI.display_tool_calls(message, response=tool_response)

        if message.content:
            content = message.content if isinstance(message.content, str) else str(message.content)
            ChatUI.display_ai_message_with_costs(message)

def show_messages_in_UI(messages: List[HumanMessage | AIMessage | SystemMessage]) -> None:
   for msg in messages:
        show_message(msg, messages)
            
st.set_page_config(
    page_title="LLM Chatbot", 
    page_icon="💬", 
    layout="centered"
)
st.title("💬 LLM Chatbot")
st.caption("Chat with your AI assistant")

if "langchain_provider" not in st.session_state:
    # Initialize LangChain provider
    st.session_state.langchain_provider = LangChainProvider()
    st.session_state.langchain_provider.set_system_prompt(Config.DEFAULT_SYSTEM_PROMPT)

provider, model, system_prompt, clear_pressed = Sidebar.render()

if clear_pressed:
    st.session_state.langchain_provider.clear_history()
    st.rerun()

st.session_state.langchain_provider.set_system_prompt(system_prompt)

display_messages: List[HumanMessage | AIMessage | SystemMessage] = st.session_state.langchain_provider.get_messages()

show_messages_in_UI(display_messages)

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
            all_messages = st.session_state.langchain_provider.get_messages()
            
            last_human_idx = -1
            for i in range(len(all_messages) - 1, -1, -1):
                if isinstance(all_messages[i], HumanMessage):
                    last_human_idx = i
                    break
            
            messages_to_display = all_messages[last_human_idx:] if last_human_idx != -1 else []
            
            with placeholder.container():
                for msg in messages_to_display:
                    show_message(msg, all_messages)
        
        st.rerun()
        
    except Exception as e:
        ChatUI.display_error(e, show_details=True)