# workflow.py
from diagrams import Diagram, Cluster, Edge
from diagrams.custom import Custom
from diagrams.onprem.client import Users
from diagrams.programming.language import Python

with Diagram("Groq-Powered FAISS RAG Pipeline with MMR", direction="LR", show=False, filename="faiss_rag_workflow", graph_attr={"splines": "spline", "nodesep": "1.0", "ranksep": "1.2"}):
    user = Users("User")
    question_box = Custom("Ask Question", "./icons/prompt.png")
    streamlit = Custom("Streamlit UI", "./icons/streamlit.png")

    with Cluster("PDF Ingestion"):
        uploader = Custom("Upload PDFs", "./icons/uploaddocs.png")
        pdf_loader = Custom("PDF Loader", "./icons/pdfloader.png")
        splitter = Python("Splitter")
        embedder = Python("Embedder")
        faiss_db = Custom("FAISS Vector DB", "./icons/faiss.png")

        uploader >> Edge(color="blue", label="1A. Load", fontcolor="blue") >> pdf_loader \
                >> Edge(color="orange", label="1B. Split", fontcolor="orange") >> splitter \
                >> Edge(color="green", label="1C. Embed", fontcolor="green") >> embedder \
                >> Edge(color="purple", label="1D. Store", fontcolor="purple") >> faiss_db

    with Cluster("RAG + LangServe"):
        mmr_retriever = Custom("MMR Retrieval", "./icons/mmr.png")
        retriever = Custom("Retriever\n(context+prompt)", "./icons/langchain.png")

        with Cluster("Backend API", direction="TB"):
            langserve = Custom("LangServe", "./icons/langserve.png")
            groq_llm = Custom("Groq LLM", "./icons/groq.png")

        answer = Custom("Answer", "./icons/answer.png")

        faiss_db >> Edge(color="purple", label="2. Retrieve with MMR", fontcolor="purple") >> mmr_retriever \
                 >> Edge(color="purple", label="2B. Deduplicate + Re-rank", fontcolor="purple") >> retriever

        retriever >> Edge(color="black", label="3. Invoke API", fontcolor="black") >> langserve 
        langserve >> Edge(color="green", label="4. Call LLM", fontcolor="green") >> groq_llm
        groq_llm >> Edge(color="black", label="5. Respond", fontcolor="black") >> answer
        answer >> Edge(color="black", label="6. Return to LangServe", fontcolor="black") >> langserve

    user >> Edge(color="red", label="A. Access GPT UI", fontcolor="red") >> question_box
    question_box >> Edge(color="red", label="B. Prompt Input", fontcolor="red") >> streamlit
    streamlit >> Edge(color="blue", label="C. Upload PDFs", fontcolor="blue") >> uploader
    streamlit >> Edge(color="blue", label="D. Prompt to Retriever", fontcolor="blue") >> retriever
    streamlit >> Edge(color="red", label="E. Final Answer", fontcolor="red") >> user
    langserve >> Edge(color="black", label="F. Return to UI", fontcolor="black") >> streamlit
