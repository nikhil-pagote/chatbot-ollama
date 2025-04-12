from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import os

# ✅ Fix for Chroma v0.4+ - required environment variables
os.environ["CHROMA_TENANT"] = "default_tenant"
os.environ["CHROMA_DATABASE"] = "default_db"

CHROMA_PATH = "chroma_db"

def load_and_embed_pdfs(upload_folder: str = "rag_uploads"):
    all_documents = []
    pdf_files = list(Path(upload_folder).glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF(s) in {upload_folder}")

    for pdf_path in pdf_files:
        loader = PyMuPDFLoader(str(pdf_path))
        docs = loader.load()
        all_documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(all_documents)

    embeddings = OllamaEmbeddings(model="llama3", base_url="http://ollama:11434")
    vectordb = Chroma.from_documents(split_docs, embedding=embeddings, persist_directory=CHROMA_PATH)

    vectordb.persist()
    print("✅ Documents embedded and saved to ChromaDB.")
