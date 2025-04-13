from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from pathlib import Path
import os

FAISS_PATH = "faiss_index"

# Set your Groq API key and model
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "mixtral-8x7b")  # Can be replaced with llama3 if supported


def load_and_embed_pdfs(upload_folder: str = "rag_uploads"):
    all_documents = []
    pdf_files = list(Path(upload_folder).glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF(s) in {upload_folder}")

    for pdf_path in pdf_files:
        loader = PyMuPDFLoader(str(pdf_path))
        docs = loader.load()

        for i, doc in enumerate(docs):
            doc.metadata["source"] = str(pdf_path)
            doc.metadata["page"] = i

        all_documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(all_documents)

    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL
    )

    faiss_db = FAISS.from_documents(split_docs, embedding=embeddings)
    faiss_db.save_local(FAISS_PATH)

    print("✅ Documents embedded and saved to FAISS using Groq model.")
