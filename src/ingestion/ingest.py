import os
import sys
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

# Config
DATA_FOLDER   = "./data"
CHROMA_PATH   = "./chroma_db"
COLLECTION    = "rag_docs"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150


# ---------------------------
# Step 1: Load PDFs
# ---------------------------
def load_documents(folder):
    docs = []
    files = [f for f in os.listdir(folder) if f.endswith(".pdf")]

    if not files:
        print("❌ No PDF files found in data/ folder!")
        sys.exit(1)

    for filename in files:
        path = os.path.join(folder, filename)
        print(f"  Loading: {filename}")
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    return docs


# ---------------------------
# Step 2: Split into chunks
# ---------------------------
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    return splitter.split_documents(docs)


# ---------------------------
# Step 3: Store in ChromaDB
# ---------------------------
def store_in_chromadb(chunks):
    print("  Using FREE HuggingFace embeddings (no API)...")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION
    )

    return vectorstore


# ---------------------------
# Main Pipeline
# ---------------------------
def main():
    print("=" * 45)
    print("  RAG Ingestion Pipeline")
    print("=" * 45)

    print("\nStep 1: Loading PDFs from data/ folder...")
    docs = load_documents(DATA_FOLDER)
    print(f"  Loaded {len(docs)} pages total")

    print("\nStep 2: Splitting into chunks...")
    chunks = split_documents(docs)
    print(f"  Created {len(chunks)} chunks")

    if chunks:
        print(f"  Example chunk preview:")
        print(f"  '{chunks[0].page_content[:120]}...'")

    print("\nStep 3: Embedding and storing in ChromaDB...")
    print("  (This may take 20–60 seconds depending on size)")
    store_in_chromadb(chunks)

    print(f"  Stored {len(chunks)} chunks in ChromaDB")

    print("\n" + "=" * 45)
    print("  ✅ Ingestion complete! Ready for Step 3.")
    print("=" * 45)


# Entry point
if __name__ == "__main__":
    main()