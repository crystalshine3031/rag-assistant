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

app = FastAPI(
    title="RAG Knowledge Assistant",
    description="Ask questions about your documents",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

print("Loading RAG chain...")
vectorstore = load_vectorstore()
chain, retriever = build_rag_chain(vectorstore)
print("RAG chain ready!")

class QuestionRequest(BaseModel):
    question: str

class SourceModel(BaseModel):
    page: int
    preview: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceModel]

@app.get("/")
def root():
    return {
        "message": "RAG Assistant is running!",
        "usage": "POST /ask with {question: 'your question'}"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "llama-3.1-8b-instant"}

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        answer = chain.invoke(request.question)
        sources = retriever.invoke(request.question)
        source_list = [
            SourceModel(
                page=int(doc.metadata.get("page", 0)) + 1,
                preview=doc.page_content[:150].replace("\n", " ")
            )
            for doc in sources
        ]
        return AnswerResponse(
            question=request.question,
            answer=answer,
            sources=source_list
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))