# serve.py (LangServe-compatible version with Upload Endpoint)
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnableLambda
from langserve import add_routes
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the PDF loader function
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.langserve_app.pdf_loader import load_and_embed_pdfs

# Load environment variables from .env file
load_dotenv()

# Configuration
HF_API_KEY = os.getenv("HF_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
BASE_DIR = Path(__file__).resolve().parent
FAISS_PATH = BASE_DIR / "faiss_index"
UPLOAD_DIR = BASE_DIR / "rag_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI()

# Load vector store with Hugging Face embeddings
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_API_KEY,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize vectordb
try:
    if Path(f"{FAISS_PATH}/index.faiss").exists() and Path(f"{FAISS_PATH}/index.pkl").exists():
        vectordb = FAISS.load_local(str(FAISS_PATH), embeddings, allow_dangerous_deserialization=True)
        logger.info("✅ FAISS index loaded successfully")
    else:
        logger.info("🔄 FAISS index not found. Initializing from scratch...")
        vectordb = None
        load_and_embed_pdfs(str(UPLOAD_DIR))
except Exception as e:
    vectordb = None
    logger.error(f"⚠️ Failed to load FAISS: {e}")

# Prompt Template
prompt_template = PromptTemplate.from_template(
    """Use the following context to answer the question.

Context:
{context}

Question:
{question}"""
)

# LLM via Groq
llm = ChatGroq(temperature=0.2, model_name=GROQ_MODEL, api_key=GROQ_API_KEY)

# Context retrieval function
def get_context(question: str, k: int = 4) -> str:
    try:
        if vectordb is None:
            return "Vector store is not ready. Please upload PDFs first."
        docs = vectordb.similarity_search(question, k=k)
        return "\n---\n".join([doc.page_content for doc in docs])
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return f"Error retrieving context: {str(e)}"

# Chain definition
def create_chain():
    def process_input(input_dict: Dict[str, Any]) -> Dict[str, str]:
        try:
            logger.info(f"Processing input: {input_dict}")
            question = input_dict.get("question", "")
            if not question:
                raise ValueError("Question field is required")
            
            context = get_context(question)
            logger.info("Context retrieved successfully")
            
            return {
                "context": context,
                "question": question
            }
        except Exception as e:
            logger.error(f"Error in process_input: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def format_output(llm_output: Any) -> Dict[str, Any]:
        try:
            logger.info("Formatting output")
            return {
                "content": llm_output.content if hasattr(llm_output, "content") else str(llm_output),
                "context": get_context(llm_output.input["question"]) if hasattr(llm_output, "input") else ""
            }
        except Exception as e:
            logger.error(f"Error in format_output: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return RunnableLambda(process_input) | prompt_template | llm | RunnableLambda(format_output)

# Add LangServe endpoint
add_routes(app, create_chain(), path="/chat")

# Endpoint for live PDF upload
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            f.write(await file.read())

        loader = PyMuPDFLoader(str(file_path))
        docs = loader.load()
        for i, doc in enumerate(docs):
            doc.metadata["source"] = file.filename
            doc.metadata["page"] = i

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = splitter.split_documents(docs)

        if not split_docs:
            return JSONResponse(status_code=400, content={"error": "No valid text chunks found in the document."})

        new_faiss = FAISS.from_documents(split_docs, embedding=embeddings)
        new_faiss.save_local(FAISS_PATH)

        return {"status": f"✅ Successfully embedded and stored '{file.filename}' in FAISS."}
    except Exception as e:
        logger.error(f"Error in upload_pdf: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
