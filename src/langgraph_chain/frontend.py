import streamlit as st
import requests

# --- CONFIG ---
API_CHAT_URL = "http://localhost:8001/chat/invoke"  # LangGraph via LangServe
API_UPLOAD_URL = "http://localhost:8001/upload"

st.set_page_config(page_title="LangGraph RAG Chatbot", layout="wide")
st.title("🤖 LangGraph RAG Chatbot")

# --- SIDEBAR ---
st.sidebar.title("📁 Upload PDFs")
pdf_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if pdf_files:
    for file in pdf_files:
        with st.spinner(f"Uploading {file.name}..."):
            res = requests.post(API_UPLOAD_URL, files={"file": (file.name, file.read(), "application/pdf")})
            if res.ok:
                st.sidebar.success(f"Uploaded: {file.name}")
            else:
                st.sidebar.error(f"Upload failed: {res.text}")

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.history = []
    st.rerun()

# --- SESSION STATE ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- CHAT ---
if prompt := st.chat_input("Ask a question..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        payload = {
            "input": {
                "question": prompt,
                "context": "",
                "answer": ""
            }
        }

        res = requests.post(API_CHAT_URL, json=payload)
        res.raise_for_status()

        data = res.json()
        output = data.get("output", {})
        reply = output.get("answer", "⚠️ No reply content.")
        context = output.get("context", None)

    except Exception as e:
        reply = f"❌ Error: {str(e)}"
        context = None

    st.session_state.history.append((prompt, reply, context))

# --- CHAT HISTORY ---
for user_msg, bot_reply, ctx in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
        if ctx:
            with st.expander("📄 Show Retrieved Context"):
                st.markdown(ctx.replace("\n", "<br>"), unsafe_allow_html=True)
                if "metadata" in reply:
                    st.json(reply["metadata"])
