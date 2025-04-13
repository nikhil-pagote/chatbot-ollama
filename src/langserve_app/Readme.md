#### pdf_loader.py	- Loads uploaded PDFs
- Splits text into chunks
- Generates embeddings using HuggingFaceInferenceAPIEmbeddings (Groq)
- Saves to FAISS vector store
#### serve.py	- Main UI via Streamlit
- Handles chat flow, file upload, filtering, score display
- Loads FAISS index and shows results