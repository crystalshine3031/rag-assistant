import os
from dotenv import load_dotenv
import chromadb

# LLM (Groq)
from langchain_groq import ChatGroq

# Embeddings (FREE alternative)
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
print(f"API Key loaded: {api_key[:8]}...")

# ---------------------------
# Test 1: Groq LLM
# ---------------------------
print("\nTesting Groq LLM...")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=api_key
)

response = llm.invoke("Say exactly: RAG setup with Groq successful")
print("Groq LLM:", response.content)

# ---------------------------
# Test 2: Embeddings (FREE)
# ---------------------------
print("\nTesting Embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector = embeddings.embed_query("hello world")
print(f"Embeddings working — vector size: {len(vector)}")

# ---------------------------
# Test 3: ChromaDB
# ---------------------------
print("\nTesting ChromaDB...")

chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection("test")

print("ChromaDB: connected")

print("\n🔥 All 3 working. Ready for Step 2!")