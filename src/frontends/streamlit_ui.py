import streamlit as st
import requests
import os
import sys
from pathlib import Path
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.chatbot_ollama.pdf_loader import load_and_embed_pdfs
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

st.set_page_config(page_title="Ollama Chat", page_icon="💬", layout="wide")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
AVAILABLE_MODELS = ["llama3", "mistral", "gemma"]
AVAILABLE_STORES = ["faiss (recommended)", "chroma", "hybrid"]

# Sidebar settings
st.sidebar.title("⚙️ Settings")
OLLAMA_MODEL = st.sidebar.selectbox("Choose a model", AVAILABLE_MODELS, index=0)
VECTOR_STORE_LABEL = st.sidebar.selectbox("Vector Store", AVAILABLE_STORES, index=0)
VECTOR_STORE = VECTOR_STORE_LABEL.split()[0]  # get 'faiss', 'chroma', or 'hybrid'
SHOW_SCORES = st.sidebar.checkbox("Show similarity scores")

# Dynamically extract filenames from vector store metadata
@st.cache_data(show_spinner=False)
def extract_filenames_from_store():
    embeddings = OllamaEmbeddings(model="llama3", base_url="http://ollama:11434")
    filenames = set()

    try:
        if VECTOR_STORE in ["chroma", "hybrid"]:
            chroma_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
            for doc in chroma_db.similarity_search("metadata probe", k=50):
                if source := doc.metadata.get("source"):
                    filenames.add(source)

        if VECTOR_STORE in ["faiss", "hybrid"]:
            faiss_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            for doc in faiss_db.similarity_search("metadata probe", k=50):
                if source := doc.metadata.get("source"):
                    filenames.add(source)
    except Exception:
        pass

    return sorted(filenames)

all_filenames = extract_filenames_from_store()
FILTERED_FILENAMES = st.sidebar.multiselect("Filter by filename (metadata)", options=all_filenames, key="filename_filter")

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
    chroma_docs, faiss_docs = [], []

    if VECTOR_STORE in ["chroma", "hybrid"]:
        chroma_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
        chroma_docs = chroma_db.similarity_search_with_score(question, k=k)
    if VECTOR_STORE in ["faiss", "hybrid"]:
        faiss_path = "faiss_index"
        if not os.path.exists(faiss_path):
            raise FileNotFoundError("FAISS index not found. Please upload PDFs to generate it first.")
        faiss_db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        faiss_docs = faiss_db.similarity_search_with_score(question, k=k)

    combined_docs = chroma_docs + faiss_docs if VECTOR_STORE == "hybrid" else chroma_docs or faiss_docs
    combined_docs.sort(key=lambda x: x[1])  # Sort by score (lower is better)

    # Filter by selected filenames
    if FILTERED_FILENAMES:
        combined_docs = [item for item in combined_docs if item[0].metadata.get("source") in FILTERED_FILENAMES]

    context_chunks = []
    for i, (doc, score) in enumerate(combined_docs[:k]):
        meta = doc.metadata
        chunk = f"<div style='border-left: 4px solid #bbb; padding-left: 1em; margin-bottom: 1em;'>"
        chunk += f"<div style='color:#6c63ff;'><strong>Chunk {i+1}</strong></div>"
        if SHOW_SCORES:
            chunk += f"<div style='color:#1f77b4;'><strong>Score:</strong> {score:.4f}</div>"
        if meta:
            chunk += f"<div style='color:#2ca02c;'><strong>File:</strong> {meta.get('source', 'unknown')}</div>"
            chunk += f"<div style='color:#d62728;'><strong>Page:</strong> {meta.get('page', '?')}</div>"
        chunk += f"<div style='margin-top: 0.5em;'>{doc.page_content}</div></div>"
        context_chunks.append(chunk)

    return "\n".join(context_chunks)

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # with st.chat_message("user"):
    #     st.markdown(prompt)

    try:
        context = get_relevant_context(prompt)
        with st.expander("📄 View Retrieved Context"):
            st.markdown(context, unsafe_allow_html=True)

        full_prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {prompt}"
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