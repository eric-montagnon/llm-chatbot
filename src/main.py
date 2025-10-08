import streamlit as st

from modules.chat import ChatManager
from modules.config import Config
from modules.ui import ChatUI, RawMessageViewer, Sidebar

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
    # Store tool results associated with message indices
    st.session_state.tool_results_map = {}

provider, model, system_prompt, stream, clear_pressed = Sidebar.render()

if clear_pressed:
    st.session_state.chat_manager.clear_chat()
    st.session_state.tool_results_map = {}  # Clear tool results too
    st.rerun()

st.session_state.chat_manager.update_system_prompt(system_prompt)

# Create two columns for chat and raw view
chat_col, raw_col = st.columns([1, 1], gap="medium")

with chat_col:
    st.header("📨 Chat Interface")
    
    # Display existing messages
    display_messages = st.session_state.chat_manager.get_display_messages()
    for idx, msg in enumerate(display_messages):
        if msg["role"] == "assistant":
            # For assistant messages, check if there are tool results to display
            with st.chat_message("assistant"):
                # Display tool results first if they exist
                if idx in st.session_state.tool_results_map:
                    tool_results = st.session_state.tool_results_map[idx]
                    if tool_results:
                        ChatUI.display_tool_calls(tool_results, in_chat_context=False)
                # Then display the message content
                st.markdown(msg["content"])
        else:
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
                    tool_results_displayed = False
                    captured_tool_results = None
                    
                    # Show streaming status in raw column
                    with raw_col:
                        status_placeholder = RawMessageViewer.display_streaming_status(0)
                    
                    for chunk_content, raw_chunk in response_gen:
                        accumulated += chunk_content
                        chunk_count += 1
                        placeholder.markdown(accumulated)
                        
                        # Check for tool results in streaming chunks
                        if not tool_results_displayed and raw_chunk.get("tool_results"):
                            tool_results = raw_chunk.get("tool_results")
                            if tool_results:
                                # Capture tool results for later storage (but don't display inline)
                                captured_tool_results = tool_results
                                tool_results_displayed = True
                        
                        # Update streaming status
                        with raw_col:
                            status_placeholder.info(f"🔄 Streaming... ({chunk_count} chunks received)")
                    
                    # Clear streaming status
                    with raw_col:
                        status_placeholder.empty()
                    
                    content = accumulated
                    
                    # Store tool results if we found any
                    if captured_tool_results:
                        # Get the index where the assistant message will be stored
                        assistant_msg_idx = len(st.session_state.chat_manager.get_display_messages())
                        st.session_state.tool_results_map[assistant_msg_idx] = captured_tool_results
                else:
                    content, raw_response = st.session_state.chat_manager.generate_response_with_raw(
                        provider_name=provider,
                        model=model,
                        stream=False
                    )
                    
                    # Check for tool results in non-streaming response
                    tool_results = raw_response.get("tool_results")
                    if tool_results:
                        # Store tool results for persistence (but don't display inline)
                        assistant_msg_idx = len(st.session_state.chat_manager.get_display_messages())
                        st.session_state.tool_results_map[assistant_msg_idx] = tool_results
                    
                    ChatUI.display_response(content)
                
                st.session_state.chat_manager.add_message("assistant", content)
                
                # Force rerun to display the message from history with tool calls
                st.rerun()
                
            except Exception as e:
                ChatUI.display_error(e, show_details=True)

with raw_col:
    # Display raw request data
    interactions = st.session_state.chat_manager.get_raw_interactions()
    RawMessageViewer.display_raw_interactions(interactions)
