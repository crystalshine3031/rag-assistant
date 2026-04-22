from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Load embeddings
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# Load existing DB
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

# Create retriever
retriever = db.as_retriever()

# Load LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Ask question
query = "What is this document about?"

docs = retriever.invoke(query)

# Combine context
context = "\n\n".join([doc.page_content for doc in docs])

prompt = f"""
Answer the question based only on the context below:

{context}

Question: {query}
"""

response = llm.invoke(prompt)

print("\n📌 ANSWER:\n")
print(response.content)