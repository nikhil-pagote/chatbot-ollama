from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from pathlib import Path
import os
from dotenv import load_dotenv

# Constants
FAISS_PATH = "faiss_index"

# Load environment variables (used for HF_API_KEY)
load_dotenv()

def load_and_embed_pdfs(upload_folder: str = "rag_uploads"):
    """
    Loads PDFs from the specified folder, extracts and splits their text content,
    generates embeddings using Hugging Face API, and saves them in a FAISS index.

    Args:
        upload_folder (str): Path to the folder containing PDF files.
    """
    pdf_files = list(Path(upload_folder).glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF(s) in {upload_folder}")

    all_documents = []
    for pdf_path in pdf_files:
        loader = PyMuPDFLoader(str(pdf_path))
        docs = loader.load()

        # Add metadata to each page
        for i, doc in enumerate(docs):
            doc.metadata["source"] = str(pdf_path.name)
            doc.metadata["page"] = i

        all_documents.extend(docs)

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(all_documents)

    # Initialize Hugging Face Embeddings
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.getenv("HF_API_KEY"),
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create and persist FAISS index
    faiss_db = FAISS.from_documents(split_docs, embedding=embeddings)
    faiss_db.save_local(FAISS_PATH)

    print("✅ FAISS index built with Hugging Face embeddings.")
