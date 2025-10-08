import streamlit as st

from chat_manager import ChatManager
from config import Config
from ui_components import ChatUI, Sidebar

st.set_page_config(
    page_title="LLM Chatbot (Streamlit)", 
    page_icon="💬", 
    layout="centered"
)
st.title("💬 Minimal LLM Chatbot")
st.caption("Switch between OpenAI and Mistral. Your messages persist during the session.")

if "chat_manager" not in st.session_state:
    st.session_state.chat_manager = ChatManager()
    st.session_state.chat_manager.update_system_prompt(Config.DEFAULT_SYSTEM_PROMPT)

provider, model, system_prompt, stream, clear_pressed = Sidebar.render()

if clear_pressed:
    st.session_state.chat_manager.clear_chat()
    st.rerun()

st.session_state.chat_manager.update_system_prompt(system_prompt)

for msg in st.session_state.chat_manager.get_display_messages():
    ChatUI.display_message(msg["role"], msg["content"])

user_input = st.chat_input("Type your message…")

if user_input:
    st.session_state.chat_manager.add_message("user", user_input)
    ChatUI.display_message("user", user_input)
    
    with st.chat_message("assistant"):
        try:
            if stream:
                response = st.session_state.chat_manager.generate_response(
                    provider_name=provider,
                    model=model,
                    stream=True
                )
                content = ChatUI.display_streaming_response(response)
            else:
                response = st.session_state.chat_manager.generate_response(
                    provider_name=provider,
                    model=model,
                    stream=False
                )
                content = ChatUI.display_response(response)
            
            st.session_state.chat_manager.add_message("assistant", content)
            
        except Exception as e:
            ChatUI.display_error(e, show_details=True)
