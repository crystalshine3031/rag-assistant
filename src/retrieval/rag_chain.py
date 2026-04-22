import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CHROMA_PATH = "./chroma_db"
COLLECTION  = "rag_docs"

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION
    )
    return vectorstore

def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
        max_tokens=512
    )

    prompt = PromptTemplate.from_template("""
You are a smart and friendly AI assistant.

Use the context to answer the question.

STRICT FORMAT (follow exactly):

1. Short Answer (2-3 lines max)

2. Key Points (bullet points)
- point 1
- point 2
- point 3

3. Simple Explanation (very easy words)

4. Example (real-life example if possible)

RULES:
- Keep it clean and readable
- Use bullet points
- Do NOT write long paragraphs
- Use context first, then add explanation

Context:
{context}

Question:
{question}

Answer:
""")

    def format_docs(docs):
        return "\n\n".join([
            f"[Source {i+1}: Page {doc.metadata.get('page', '?')+1}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
        ])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask(question: str):
    print(f"\nQuestion: {question}")
    print("-" * 45)

    vectorstore = load_vectorstore()
    chain, retriever = build_rag_chain(vectorstore)

    # Get source chunks
    sources = retriever.invoke(question)

    # Get answer
    answer = chain.invoke(question)

    print(f"Answer: {answer}")
    print("\nSources used:")
    for i, doc in enumerate(sources):
        page = doc.metadata.get('page', '?')
        preview = doc.page_content[:80].replace('\n', ' ')
        print(f"  [{i+1}] Page {page+1}: {preview}...")

    return answer, sources


if __name__ == "__main__":
    print("=" * 45)
    print("  RAG Chain Test")
    print("=" * 45)

    test_questions = [
        "What is machine learning?",
        "What are the types of machine learning?",
        "What is supervised learning?"
    ]

    for question in test_questions:
        ask(question)
        print()