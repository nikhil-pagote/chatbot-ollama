# langserve_api.py
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langserve import add_routes
import logging

sys.path.append(str(Path(__file__).resolve().parents[1]))
from langgraph_chain.basic_graph import rag_graph_app  # Import LangGraph chain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "rag_uploads"
FAISS_PATH = BASE_DIR / "faiss_index"

UPLOAD_DIR.mkdir(exist_ok=True)

# FastAPI app
app = FastAPI()

# Embeddings instance (for live upload endpoint)
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_API_KEY,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# PDF upload endpoint for embedding new files
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Load and split
        loader = PyMuPDFLoader(str(file_path))
        docs = loader.load()

        for i, doc in enumerate(docs):
            doc.metadata["source"] = file.filename
            doc.metadata["page"] = i

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = splitter.split_documents(docs)

        if not split_docs:
            return JSONResponse(status_code=400, content={"error": "No valid text chunks found."})

        print(f"📎 Number of chunks created: {len(split_docs)}")
        print(f"📄 Sample chunk: {split_docs[0].page_content[:300]}")

        # Load embeddings
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=HF_API_KEY,
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Merge or create FAISS index
        if (FAISS_PATH / "index.faiss").exists():
            existing = FAISS.load_local(str(FAISS_PATH), embeddings, allow_dangerous_deserialization=True)
            existing.merge_from(FAISS.from_documents(split_docs, embedding=embeddings))
            existing.save_local(str(FAISS_PATH))
            print("✅ Merged into existing FAISS index.")
        else:
            new = FAISS.from_documents(split_docs, embedding=embeddings)
            new.save_local(str(FAISS_PATH))
            print("✅ Created new FAISS index.")

        return {"status": f"✅ Successfully embedded and stored '{file.filename}' in FAISS."}

    except Exception as e:
        print(f"❌ Upload error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# Add LangGraph-based RAG pipeline to LangServe
add_routes(app, rag_graph_app, path="/chat")
