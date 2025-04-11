import streamlit as st
import requests
import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

st.set_page_config(page_title="Ollama Chat", page_icon="💬")
st.title("💬 Ollama Chatbot")

# Session history
if "history" not in st.session_state:
    st.session_state.history = []

# Input box
with st.form("chat-form", clear_on_submit=True):
    user_input = st.text_input("Ask me anything:", "")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    try:
        res = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": user_input,
            "stream": False
        })
        res.raise_for_status()
        result = res.json()["response"]
    except Exception as e:
        result = f"Error: {str(e)}"

    st.session_state.history.append((user_input, result))

# Show conversation
for user, bot in reversed(st.session_state.history):
    st.markdown(f"**You:** {user}")
    st.markdown(f"**Bot:** {bot}")
    st.markdown("---")
