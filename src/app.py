import os
from dataclasses import dataclass
from typing import List, Literal, Dict

import streamlit as st
from dotenv import load_dotenv

# ---- Load env ----
load_dotenv()

# ---- Providers ----
Provider = Literal["OpenAI", "Mistral"]


@dataclass
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


def get_openai_client():
    # OpenAI >= 1.0 style client
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    return OpenAI(api_key=api_key)


def get_mistral_client():
    # mistralai >= 1.0
    from mistralai import Mistral
    api_key = os.getenv("MISTRAL_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing MISTRAL_API_KEY")
    return Mistral(api_key=api_key)


def call_openai(messages: List[ChatMessage], model: str, stream: bool = True):
    client = get_openai_client()
    # Convert to OpenAI format
    formatted = [{"role": m.role, "content": m.content} for m in messages]
    if stream:
        return client.chat.completions.create(model=model, messages=formatted, stream=True)
    else:
        return client.chat.completions.create(model=model, messages=formatted)


def call_mistral(messages: List[ChatMessage], model: str, stream: bool = True):
    client = get_mistral_client()
    formatted = [{"role": m.role, "content": m.content} for m in messages]
    if stream:
        # stream=True returns a generator of events
        return client.chat.stream(model=model, messages=formatted)
    else:
        return client.chat.complete(model=model, messages=formatted)


# ---- UI ----
st.set_page_config(page_title="LLM Chatbot (Streamlit)", page_icon="💬", layout="centered")
st.title("💬 Minimal LLM Chatbot")
st.caption("Switch between OpenAI and Mistral. Your messages persist during the session.")

with st.sidebar:
    st.header("Settings")
    provider: Provider = st.selectbox("Provider", ["OpenAI", "Mistral"])
    # You can type any model name you have access to
    default_openai = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    default_mistral = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    model = st.text_input("Model", value=default_openai if provider == "OpenAI" else default_mistral)
    system_prompt = st.text_area(
        "System prompt",
        value="You are a helpful assistant.",
        height=80,
    )
    stream = st.toggle("Stream responses", value=True)
    if st.button("Clear chat"):
        st.session_state.messages = []

# ---- Session state ----
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

# Keep system prompt in sync if user changes it mid-session
if st.session_state.messages and st.session_state.messages[0]["role"] == "system":
    st.session_state.messages[0]["content"] = system_prompt

# ---- Chat history render ----
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue  # don't render system
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- Input ----
user_input = st.chat_input("Type your message…")


def add_message(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content})

if user_input:
    add_message("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            if provider == "OpenAI":
                # OpenAI streaming
                if stream:
                    resp_stream = call_openai([ChatMessage(**m) for m in st.session_state.messages], model, stream=True)
                    placeholder = st.empty()
                    acc = ""
                    for event in resp_stream:
                        delta = event.choices[0].delta.content or ""
                        acc += delta
                        placeholder.markdown(acc)
                    add_message("assistant", acc)
                else:
                    resp = call_openai([ChatMessage(**m) for m in st.session_state.messages], model, stream=False)
                    text = resp.choices[0].message.content
                    st.markdown(text)
                    add_message("assistant", text)

            elif provider == "Mistral":
                if stream:
                    acc = ""
                    placeholder = st.empty()
                    for event in call_mistral([ChatMessage(**m) for m in st.session_state.messages], model, stream=True):
                        # mistral stream events contain .data.choices[0].delta.content
                        if hasattr(event, "data") and event.data and event.data.choices:
                            delta = event.data.choices[0].delta.get("content") or ""
                            acc += delta
                            placeholder.markdown(acc)
                    add_message("assistant", acc)
                else:
                    resp = call_mistral([ChatMessage(**m) for m in st.session_state.messages], model, stream=False)
                    text = resp.choices[0].message.get("content", "")
                    st.markdown(text)
                    add_message("assistant", text)

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
