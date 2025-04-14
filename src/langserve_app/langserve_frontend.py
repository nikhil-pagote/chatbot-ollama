import streamlit as st
import requests
from pathlib import Path

# --- CONFIG ---
API_CHAT_URL = "http://localhost:8001/chat/invoke"
API_UPLOAD_URL = "http://localhost:8001/upload"
AVAILABLE_MODELS = ["llama3-8b-8192", "mixtral-8x7b", "gemma-7b"]

# --- PAGE SETUP ---
st.set_page_config(page_title="Groq RAG Chatbot via LangServe", layout="wide")
st.title("💬 Groq RAG Chatbot (LangServe)")

# --- SIDEBAR ---
st.sidebar.title("⚙️ Settings")
st.sidebar.selectbox("Model (from .env)", AVAILABLE_MODELS, index=0, disabled=True)
SHOW_CONTEXT = st.sidebar.checkbox("Show retrieved context")

st.sidebar.markdown("### 📎 Upload PDFs to LangServe")
pdf_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

# Upload PDF files to LangServe backend
if pdf_files:
    for file in pdf_files:
        with st.spinner(f"📤 Uploading: {file.name}"):
            res = requests.post(
                API_UPLOAD_URL,
                files={"file": (file.name, file.read(), "application/pdf")}
            )
            if res.status_code == 200:
                st.sidebar.success(f"✅ Uploaded: {file.name}")
            else:
                st.sidebar.error(f"❌ Failed: {file.name} - {res.text}")

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.history = []
    st.rerun()

# --- SESSION STATE ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- CHAT UI ---
if prompt := st.chat_input("Ask me anything..."):
    # with st.chat_message("user"):
    #     st.markdown(prompt)

    try:
        payload = {"input": {"question": prompt}}
        print("✅ Sending request to", API_CHAT_URL)
        print("📦 Payload:", payload)

        res = requests.post(API_CHAT_URL, json=payload)
        print("📬 Response status:", res.status_code)
        print("📬 Headers:", dict(res.headers))
        print("📬 Body:", res.text)

        res.raise_for_status()
        response = res.json()

        output = response.get("output", {})
        reply = output.get("content")
        context = output.get("context")

        # Final fallback and debug
        print("🧠 Final Reply:", reply)
        print("📄 Final Context:", context)

        if not reply:
            reply = "⚠️ No reply returned."
        if not context:
            context = "⚠️ Context not available."

        st.session_state.history.append((prompt, reply, context))

    except Exception as e:
        st.session_state.history.append((prompt, f"❌ Error: {str(e)}", None))


# --- CHAT HISTORY ---
for user_msg, bot_reply, ctx in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_reply or "⚠️ No reply returned.")
        if SHOW_CONTEXT and ctx:
            st.markdown("---")
            st.markdown("**Retrieved Context:**", help="Context from FAISS DB used to answer")
            st.markdown(ctx)
