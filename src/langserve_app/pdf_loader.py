from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from pathlib import Path
import os

FAISS_PATH = "faiss_index"


def load_and_embed_pdfs(upload_folder: str = "rag_uploads"):
    pdf_files = list(Path(upload_folder).glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF(s) in {upload_folder}")

    all_documents = []
    for pdf_path in pdf_files:
        loader = PyMuPDFLoader(str(pdf_path))
        docs = loader.load()
        for i, doc in enumerate(docs):
            doc.metadata["source"] = str(pdf_path.name)
            doc.metadata["page"] = i
        all_documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(all_documents)

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://ollama:11434"
    )

    faiss_db = FAISS.from_documents(split_docs, embedding=embeddings)
    faiss_db.save_local(FAISS_PATH)

    print("✅ FAISS index built with Ollama embeddings.")
