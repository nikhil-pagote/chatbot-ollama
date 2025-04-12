import streamlit as st
import requests
import os
import sys
from pathlib import Path
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.chatbot_ollama.pdf_loader import load_and_embed_pdfs
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

st.set_page_config(page_title="Ollama Chat", page_icon="💬", layout="wide")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
AVAILABLE_MODELS = ["llama3", "mistral", "gemma"]
AVAILABLE_STORES = ["chroma", "faiss"]

# Sidebar settings
st.sidebar.title("⚙️ Settings")
OLLAMA_MODEL = st.sidebar.selectbox("Choose a model", AVAILABLE_MODELS, index=0)
VECTOR_STORE = st.sidebar.selectbox("Vector Store", AVAILABLE_STORES, index=0)

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
    st.sidebar.success("✅ PDFs processed and automatically persisted to ChromaDB and FAISS.")

st.title("💬 Ollama Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

# Load persisted Vector DB
def get_relevant_context(question: str, k: int = 3) -> str:
    embeddings = OllamaEmbeddings(model="llama3", base_url="http://ollama:11434")
    if VECTOR_STORE == "chroma":
        vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    else:
        index_path = "faiss_index"
        if not os.path.exists(index_path):
            raise FileNotFoundError("FAISS index not found. Please upload PDFs to generate it first.")
        vectordb = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    docs: list[Document] = vectordb.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        context = get_relevant_context(prompt)
        full_prompt = f"""Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {prompt}"""
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
