import os
from pathlib import Path
import logging
from typing import TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent
FAISS_PATH = BASE_DIR / "faiss_index"
logger.info(f"🧽 FAISS path being used: {FAISS_PATH}")

# Embeddings + VectorStore
HF_API_KEY = os.getenv("HF_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "deepseek-r1-distill-llama-70b")

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_API_KEY,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = None

def get_vectordb():
    global vectordb
    if vectordb is None:
        try:
            if (FAISS_PATH / "index.faiss").exists() and (FAISS_PATH / "index.pkl").exists():
                vectordb = FAISS.load_local(str(FAISS_PATH), embeddings, allow_dangerous_deserialization=True)
                logger.info(f"✅ FAISS index loaded from: {FAISS_PATH}")
                print(f"📆 FAISS index loaded. Number of vectors: {vectordb.index.ntotal}")
            else:
                logger.warning(f"⚠️ FAISS index files not found in {FAISS_PATH}")
        except Exception as e:
            logger.warning(f"⚠️ FAISS index load failed: {e}")
            vectordb = None
    return vectordb

# Define LangGraph schema using TypedDict
class RAGGraphState(TypedDict, total=False):  # total=False makes keys optional
    question: str
    context: str
    answer: str

# Define nodes
def retrieve_node(state: RAGGraphState) -> RAGGraphState:
    question = state["question"]
    db = get_vectordb()
    if db is None:
        logger.warning("⚠️ Vector store is None during retrieval.")
        return {"question": question, "context": "⚠️ No documents found.", "answer": ""}
    try:
        # Use MMR search for better context
        docs = db.max_marginal_relevance_search(question, k=4, fetch_k=10)
        logger.info(f"🔍 Retrieved {len(docs)} docs using MMR for: {question}")
        for i, doc in enumerate(docs):
            logger.info(f"📄 Doc {i}: {doc.page_content[:150]}")

        if not docs:
            return {"question": question, "context": "⚠️ No documents found.", "answer": ""}

        context = "\n---\n".join([doc.page_content for doc in docs])
        return {"question": question, "context": context, "answer": ""}

    except Exception as e:
        logger.error(f"❌ Retrieval error: {e}")
        return {"question": question, "context": f"⚠️ Retrieval failed: {e}", "answer": ""}

def answer_node(state: RAGGraphState) -> RAGGraphState:
    llm = ChatGroq(model_name=GROQ_MODEL, temperature=0.2, api_key=GROQ_API_KEY)
    prompt = f"""Use the context below to answer the question.

Context:
{state['context']}

Question:
{state['question']}"""
    try:
        response = llm.invoke(prompt)

        # Ensure we return a plain string
        if isinstance(response, BaseMessage):
            final_answer = response.content
        else:
            final_answer = str(response)

        return {**state, "answer": final_answer}

    except Exception as e:
        return {
            **state,
            "answer": f"⚠️ LLM error: {str(e)}"
        }

# Build LangGraph
builder = StateGraph(RAGGraphState)
builder.add_node("retriever", retrieve_node)
builder.add_node("llm", answer_node)
builder.set_entry_point("retriever")
builder.add_edge("retriever", "llm")
builder.add_edge("llm", END)

rag_graph_app = builder.compile()
