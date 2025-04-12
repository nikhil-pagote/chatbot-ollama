import streamlit as st
import requests
import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.chatbot_ollama.pdf_loader import load_and_embed_pdfs
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

st.set_page_config(page_title="Ollama Chat", page_icon="💬", layout="wide")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
AVAILABLE_MODELS = ["llama3", "mistral", "gemma"]

# Sidebar settings
st.sidebar.title("⚙️ Settings")
OLLAMA_MODEL = st.sidebar.selectbox("Choose a model", AVAILABLE_MODELS, index=0)

st.sidebar.markdown("---")
if st.sidebar.button("🩹 Clear Chat"):
    st.session_state.history = []
    st.rerun()

# File uploader for RAG (PDFs)
st.sidebar.markdown("### 📌 Upload PDFs for RAG")
pdf_files = st.sidebar.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

# Process PDF files for RAG
if pdf_files:
    upload_dir = Path("rag_uploads")
    upload_dir.mkdir(exist_ok=True)

    for file in pdf_files:
        filepath = upload_dir / file.name
        with open(filepath, "wb") as f:
            f.write(file.read())

    # Trigger the embedding
    with st.spinner("Processing PDFs and updating vector store..."):
        load_and_embed_pdfs(str(upload_dir))
    st.sidebar.success("✅ PDFs processed and embedded into ChromaDB.")

st.title("💬 Ollama Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

# Load persisted ChromaDB
def get_relevant_context(question: str, k: int = 3) -> str:
    vectordb = Chroma(
        persist_directory="chroma_db",
        embedding_function=OllamaEmbeddings(model="llama3", base_url="http://ollama:11434")
    )
    docs: list[Document] = vectordb.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        context = get_relevant_context(prompt)

        if context.strip():
            full_prompt = f"""
You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, simply reply: "I don't know".

### CONTEXT:
{context}

### QUESTION:
{prompt}

### ANSWER:
"""
        else:
            full_prompt = prompt  # fallback to base model knowledge

        res = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": full_prompt,
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

# Optional debug panel for retrieved context
if context := st.session_state.get("_last_context"):
    with st.expander("📄 Retrieved Context"):
        st.markdown(context)

# Store last context for debug use
if prompt:
    st.session_state["_last_context"] = context