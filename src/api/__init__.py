import os
import sys
import warnings
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ── Setup ─────────────────────────────────────────────
warnings.filterwarnings("ignore")
load_dotenv()

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.retrieval.rag_chain import load_vectorstore, build_rag_chain

# ── Global variables ──────────────────────────────────
vectorstore = None
chain = None
retriever = None

# ── Lifespan ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore, chain, retriever

    print(" Loading RAG system...")

    try:
        vectorstore = load_vectorstore()
        chain, retriever = build_rag_chain(vectorstore)
        print(" RAG system ready!")

    except Exception as e:
        print(f" Error loading RAG system: {e}")
        raise RuntimeError(e)

    yield

    print(" Shutting down RAG system...")

# ── App ───────────────────────────────────────────────
app = FastAPI(
    title="RAG Knowledge Assistant",
    description="Ask questions about your documents",
    version="1.0.0",
    lifespan=lifespan
)

# ── Middleware ────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str

class SourceModel(BaseModel):
    page: int
    preview: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceModel]

# ── Routes ────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "RAG Assistant is running 🚀",
        "usage": "POST /ask with {question: 'your question'}"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "Groq - llama-3.1-8b-instant"
    }

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    global chain, retriever

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if chain is None or retriever is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")

    try:
        answer = chain.invoke(request.question)
        docs = retriever.invoke(request.question)

        sources = [
            SourceModel(
                page=int(doc.metadata.get("page", 0)) + 1,
                preview=doc.page_content[:150].replace("\n", " ")
            )
            for doc in docs
        ]

        return AnswerResponse(
            question=request.question,
            answer=answer,
            sources=sources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))