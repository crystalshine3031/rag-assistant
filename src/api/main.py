import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from src.retrieval.rag_chain import load_vectorstore, build_rag_chain

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ONCE at startup — not on every request
vectorstore = None
rag_chain = None
retriever = None

@app.on_event("startup")
async def startup_event():
    global vectorstore, rag_chain, retriever
    print("Loading vectorstore and RAG chain...")
    vectorstore = load_vectorstore()
    rag_chain, retriever = build_rag_chain(vectorstore)
    print("Ready!")

class Question(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
async def ask(q: Question):
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="Model still loading, try again in a moment")
    try:
        sources = retriever.invoke(q.question)
        answer = rag_chain.invoke(q.question)
        return {
            "answer": answer,
            "sources": [{"page": doc.metadata.get("page", 0)} for doc in sources]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))