# serve.py
import streamlit as st
import os
import sys
from pathlib import Path
import shutil
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnableLambda

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.chatbot_ollama.pdf_loader import load_and_embed_pdfs

# --- Config ---
st.set_page_config(page_title="Groq Chat", page_icon="💬", layout="wide")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
FAISS_PATH = "faiss_index"
UPLOAD_DIR = Path("rag_uploads")

# --- Sidebar ---
st.sidebar.title("⚙️ Settings")
SHOW_SCORES = st.sidebar.checkbox("Show similarity scores")

@st.cache_data(show_spinner=False)
def get_filenames_from_faiss():
    filenames = set()
    try:
        embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=GROQ_API_KEY, model_name=GROQ_MODEL)
        faiss_db = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        for doc in faiss_db.similarity_search("metadata probe", k=50):
            if source := doc.metadata.get("source"):
                filenames.add(source)
    except Exception:
        pass
    return sorted(filenames)

FILTERED_FILENAMES = st.sidebar.multiselect("Filter by filename (metadata)", options=get_filenames_from_faiss())

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.history = []
    st.rerun()

st.sidebar.markdown("### 📎 Upload PDFs for RAG")
pdf_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if pdf_files:
    UPLOAD_DIR.mkdir(exist_ok=True)
    for file in pdf_files:
        filepath = UPLOAD_DIR / file.name
        with open(filepath, "wb") as f:
            f.write(file.read())
    with st.spinner("Embedding documents into FAISS using Groq..."):
        load_and_embed_pdfs(str(UPLOAD_DIR))
    st.sidebar.success("✅ PDFs processed and stored in FAISS.")

# --- LangChain pipeline ---
embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=GROQ_API_KEY, model_name=GROQ_MODEL)
vectordb = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

prompt_template = PromptTemplate.from_template(
    """Use the following context to answer the question.

Context:
{context}

Question:
{question}"""
)

llm = ChatGroq(temperature=0.2, model_name=GROQ_MODEL, api_key=GROQ_API_KEY)

def get_context(question: str, k: int = 4):
    docs = vectordb.similarity_search_with_score(question, k=k)
    if FILTERED_FILENAMES:
        docs = [item for item in docs if item[0].metadata.get("source") in FILTERED_FILENAMES]
    return docs

def build_chain():
    return RunnableMap({
        "context": RunnableLambda(lambda x: "\n---\n".join([doc.page_content for doc, _ in get_context(x["question"])])),
        "question": lambda x: x["question"]
    }) | prompt_template | llm

chain = build_chain()

# --- UI ---
st.title("💬 Groq RAG Chatbot")
if "history" not in st.session_state:
    st.session_state.history = []

if prompt := st.chat_input("Ask me anything..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    context_chunks = get_context(prompt)
    context_html = "".join([
        f"<div style='border-left: 4px solid #bbb; padding-left: 1em; margin-bottom: 1em;'>"
        f"<div style='color:#6c63ff;'><strong>Chunk {i+1}</strong></div>"
        f"<div style='color:#1f77b4;'><strong>Score:</strong> {score:.4f}</div>"
        f"<div style='color:#2ca02c;'><strong>File:</strong> {doc.metadata.get('source')}</div>"
        f"<div style='color:#d62728;'><strong>Page:</strong> {doc.metadata.get('page')}</div>"
        f"<div style='margin-top: 0.5em;'>{doc.page_content}</div></div>"
        for i, (doc, score) in enumerate(context_chunks)
    ])

    try:
        result = chain.invoke({"question": prompt})
    except Exception as e:
        result = f"Error: {str(e)}"

    st.session_state.history.append((prompt, result.content, context_html))

# --- History ---
for user, bot, ctx in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(user)
    with st.chat_message("assistant"):
        st.markdown(bot)
        if SHOW_SCORES:
            st.markdown("""<hr/><h4 style='margin-bottom: 0.5em;'>Retrieved Context:</h4>""", unsafe_allow_html=True)
            st.markdown(ctx, unsafe_allow_html=True)
