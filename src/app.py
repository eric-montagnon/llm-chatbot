import streamlit as st

from chat_manager import ChatManager
from config import Config
from ui_components import ChatUI, RawMessageViewer, Sidebar

st.set_page_config(
    page_title="LLM Chatbot with Raw View", 
    page_icon="💬", 
    layout="wide"  # Change to wide layout for two-column view
)
st.title("💬 LLM Chatbot with Request Viewer")
st.caption("See the exact messages being sent to the LLM")

if "chat_manager" not in st.session_state:
    st.session_state.chat_manager = ChatManager()
    st.session_state.chat_manager.update_system_prompt(Config.DEFAULT_SYSTEM_PROMPT)

provider, model, system_prompt, stream, clear_pressed = Sidebar.render()

if clear_pressed:
    st.session_state.chat_manager.clear_chat()
    st.rerun()

st.session_state.chat_manager.update_system_prompt(system_prompt)

# Create two columns for chat and raw view
chat_col, raw_col = st.columns([1, 1], gap="medium")

with chat_col:
    st.header("📨 Chat Interface")
    
    # Display existing messages
    for msg in st.session_state.chat_manager.get_display_messages():
        ChatUI.display_message(msg["role"], msg["content"])
    
    # Chat input
    user_input = st.chat_input("Type your message…")
    
    if user_input:
        st.session_state.chat_manager.add_message("user", user_input)
        ChatUI.display_message("user", user_input)
        
        with st.chat_message("assistant"):
            try:
                if stream:
                    # Use the new method that returns raw data
                    response_gen = st.session_state.chat_manager.generate_response_with_raw(
                        provider_name=provider,
                        model=model,
                        stream=True
                    )
                    
                    placeholder = st.empty()
                    accumulated = ""
                    chunk_count = 0
                    
                    # Show streaming status in raw column
                    with raw_col:
                        status_placeholder = RawMessageViewer.display_streaming_status(0)
                    
                    for chunk_content, raw_chunk in response_gen:
                        accumulated += chunk_content
                        chunk_count += 1
                        placeholder.markdown(accumulated)
                        
                        # Update streaming status
                        with raw_col:
                            status_placeholder.info(f"🔄 Streaming... ({chunk_count} chunks received)")
                    
                    # Clear streaming status
                    with raw_col:
                        status_placeholder.empty()
                    
                    content = accumulated
                else:
                    content, raw_response = st.session_state.chat_manager.generate_response_with_raw(
                        provider_name=provider,
                        model=model,
                        stream=False
                    )
                    ChatUI.display_response(content)
                
                st.session_state.chat_manager.add_message("assistant", content)
                
            except Exception as e:
                ChatUI.display_error(e, show_details=True)

with raw_col:
    # Display raw request data
    interactions = st.session_state.chat_manager.get_raw_interactions()
    RawMessageViewer.display_raw_interactions(interactions)
