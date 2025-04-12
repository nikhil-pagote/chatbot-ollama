from diagrams import Diagram, Cluster, Edge
from diagrams.custom import Custom
from diagrams.onprem.client import Users
from diagrams.programming.language import Python
from diagrams.programming.framework import React
from diagrams.programming.flowchart import InputOutput
from diagrams.onprem.mlops import Mlflow
from diagrams.generic.storage import Storage

with Diagram("Ollama RAG Chatbot Architecture", show=False, direction="TB"):

    user = Users("User")

    with Cluster("Frontend Options"):
        js_ui = React("popup.js")
        streamlit = Python("Streamlit UI")
        frontend = [js_ui, streamlit]

    with Cluster("API Server (LangServe)"):
        langserve = Python("LangServe")
        mcp = Python("MCP Integration")

    with Cluster("RAG Engine"):
        chroma = Storage("ChromaDB")
        faiss = Storage("FAISS")
        rag_stack = [chroma, faiss]

    with Cluster("LLM Backend"):
        ollama = Custom("Ollama (GPU)", "./ollama-icon.png")  # Use a custom icon if desired

    with Cluster("Data Source"):
        pdfs = InputOutput("Uploaded PDFs")

    user >> frontend >> langserve
    langserve >> mcp
    mcp >> rag_stack
    rag_stack >> ollama
    pdfs >> chroma
    langserve << Edge(color="darkgreen", style="dashed") << ollama
