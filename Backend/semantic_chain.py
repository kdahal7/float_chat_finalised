from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()

class QueryRequest(BaseModel):
    question: str

router = APIRouter()

# --- Configuration ---
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = "argo_faiss_vectorstore"

try:    
    if os.path.exists(VECTORSTORE_DIR):
        # --- Embeddings and Vector Store Setup ---
        hf_embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        vectorstore = FAISS.load_local(
            folder_path=VECTORSTORE_DIR,
            embeddings=hf_embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        # --- LLM and Prompt Setup ---
        llm = ChatOllama(
            model="llama3.1:8b",
            temperature=0.1,
            num_ctx=4096,
            num_predict=8192,  # Increased for longer responses
            top_p=0.9
        )

        custom_prompt_template = """You are FloatChat, an AI-powered oceanographic analyst specializing in Indian Ocean ARGO float data discovery and visualization.

Using the retrieved ARGO measurements, provide comprehensive answers that include:

🔍 **Data Analysis**: Specific measurements (temperature °C, salinity PSU, pressure/depth dbar)
🌊 **Oceanographic Context**: Explain patterns, water masses, and ocean processes  
📍 **Geographic Details**: Indian Ocean locations, coordinates, regional characteristics
🚢 **ARGO Float Info**: Platform numbers, measurement dates, float trajectories
📊 **Comparative Analysis**: Trends, anomalies, and data relationships

**Retrieved ARGO Data:**
{context}

**User Query:** {question}

**Professional Oceanographic Response:**"""
        PROMPT = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

        # --- The Semantic Search RAG Chain ---
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        SEMANTIC_RAG_AVAILABLE = True
        print("✅ Semantic Search RAG pipeline loaded successfully with Ollama!")
    else:
        SEMANTIC_RAG_AVAILABLE = False
        print("⚠️ Vector store not found. Semantic search endpoint will be unavailable.")

except Exception as e:
    SEMANTIC_RAG_AVAILABLE = False
    print(f"⚠️ Semantic Search RAG pipeline failed to load: {e}")


@router.post("/semantic_query")
def semantic_query(request: QueryRequest):
    if not SEMANTIC_RAG_AVAILABLE:
        raise HTTPException(status_code=500, detail="Semantic RAG pipeline is not available. Check if the vector store exists.")

    try:
        result = qa_chain.invoke({"query": request.question})
        return {
            "answer": result['result'],
            "source_documents_count": len(result['source_documents'])
        }
    except Exception as e:
        print(f"Semantic RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process query. Error: {e}")


def get_semantic_chain():
    """Return the semantic RAG chain for direct use"""
    if SEMANTIC_RAG_AVAILABLE:
        return qa_chain
    else:
        raise Exception("Semantic chain is not available - check vector store and API configuration")