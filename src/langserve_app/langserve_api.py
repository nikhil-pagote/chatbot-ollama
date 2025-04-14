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
        if not Path(f"{FAISS_PATH}/index.faiss").exists():
            return "Vector store is not ready. Please upload PDFs first."

        # Dynamically load FAISS
        vectordb = FAISS.load_local(str(FAISS_PATH), embeddings, allow_dangerous_deserialization=True)

        docs = vectordb.similarity_search(question, k=k)
        context_parts = []

        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "N/A")
            part = f"""### Chunk {i}
📄 **Source**: {source}
📄 **Page**: {page}

{doc.page_content}
"""
            context_parts.append(part)

        return "\n---\n".join(context_parts)

    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return f"Error retrieving context: {str(e)}"


# Chain definition
from langchain_core.runnables import RunnableLambda
from typing import Dict, Any
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

from langchain_core.runnables import RunnableLambda
from typing import Dict, Any
from fastapi import HTTPException
import logging

logger = logging.getLogger("langserve_api")

def create_chain():
    try:
        # Step 1: Extract and return question + context
        def process_input(input_dict: Dict[str, Any]) -> tuple[str, str]:
            question = input_dict.get("question", "")
            if not question:
                raise ValueError("Missing 'question' in input payload")
            context = get_context(question)
            logger.info(f"📥 Received input: {question}")
            logger.info(f"📄 Retrieved context:\n{context}")
            return question, context

        # Step 2: Format prompt using question and context
        def prepare_prompt(data: tuple[str, str]) -> str:
            question, context = data
            return prompt_template.format(question=question, context=context)

        # Step 3: Format LLM output + attach original context
        def format_output(llm_output: Any, original_input: tuple[str, str]) -> Dict[str, Any]:
            try:
                logger.info(f"📤 LLM Output: {llm_output}")
                return {
                    "content": getattr(llm_output, "content", str(llm_output)),
                    "context": original_input[1]  # context
                }
            except Exception as e:
                logger.error(f"🚨 Error in format_output: {e}")
                raise HTTPException(status_code=500, detail=f"Format Output Error: {str(e)}")

        # Chain assembly
        return (
            RunnableLambda(process_input)
            | RunnableLambda(lambda q_and_ctx: {
                "prompt": prompt_template.format(question=q_and_ctx[0], context=q_and_ctx[1]),
                "original_input": q_and_ctx
            })
            | RunnableLambda(lambda data: {
                "llm_output": llm.invoke(data["prompt"]),
                "original_input": data["original_input"]
            })
            | RunnableLambda(lambda data: format_output(data["llm_output"], data["original_input"]))
        )

    except Exception as e:
        logger.error(f"❌ Failed to build LangChain pipeline: {e}")
        return None


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

        if Path(f"{FAISS_PATH}/index.faiss").exists():
            existing_db = FAISS.load_local(str(FAISS_PATH), embeddings, allow_dangerous_deserialization=True)
            existing_db.add_documents(split_docs)
            existing_db.save_local(FAISS_PATH)
        else:
            new_faiss = FAISS.from_documents(split_docs, embedding=embeddings)
            new_faiss.save_local(FAISS_PATH)

        return {"status": f"✅ Successfully embedded and stored '{file.filename}' in FAISS."}
    except Exception as e:
        logger.error(f"Error in upload_pdf: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
