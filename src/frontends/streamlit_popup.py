import streamlit as st
import requests
import os
from pathlib import Path

st.set_page_config(page_title="Ollama Chat", page_icon="💬", layout="wide")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
AVAILABLE_MODELS = ["llama3", "mistral", "gemma"]

# Sidebar settings
st.sidebar.title("⚙️ Settings")
OLLAMA_MODEL = st.sidebar.selectbox("Choose a model", AVAILABLE_MODELS, index=0)

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.history = []
    st.rerun()

# File uploader for RAG (PDFs)
st.sidebar.markdown("### 📎 Upload PDFs for RAG")
pdf_files = st.sidebar.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

# Process PDF files placeholder
if pdf_files:
    for file in pdf_files:
        # Save uploaded file for future RAG indexing
        upload_dir = Path("rag_uploads")
        upload_dir.mkdir(exist_ok=True)
        filepath = upload_dir / file.name
        with open(filepath, "wb") as f:
            f.write(file.read())
    st.sidebar.success("Files uploaded. RAG pipeline integration coming soon.")

st.title("💬 Ollama Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        res = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        })
        res.raise_for_status()
        result = res.json()["response"]
    except Exception as e:
        result = f"Error: {str(e)}"

    st.session_state.history.append((prompt, result))

# Display chat history
for user, bot in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(user)
    with st.chat_message("assistant"):
        st.markdown(bot)
