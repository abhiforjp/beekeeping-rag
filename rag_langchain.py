"""
LangChain version of the same RAG pipeline (still 100% free).

Stack:
  - Splitter    : RecursiveCharacterTextSplitter
  - Embeddings  : HuggingFaceEmbeddings (all-MiniLM-L6-v2, local)
  - Vector store: InMemoryVectorStore (built into langchain-core)
  - LLM         : ChatGroq (llama-3.1-8b-instant, free tier) — needs GROQ_API_KEY

Usage:
    python rag_langchain.py
"""

import os

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "beekeeping.txt")

ASSIGNMENT_QUESTIONS = [
    "What is the minimum internal temperature for a Langstroth Hive in winter?",
    "Why are entrance reducers used?",
    "How do beekeepers control condensation?",
]

PROMPT = ChatPromptTemplate.from_template(
    """You are a precise assistant. Answer the question using ONLY the context below.
Answer in one short sentence. If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""
)


def build_vector_store() -> InMemoryVectorStore:
    """Ingest: load -> chunk -> embed -> index."""
    with open(DATA_PATH, encoding="utf-8") as f:
        text = f.read().strip()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=120,        # small chunks: the doc is only 4 sentences
        chunk_overlap=20,
        separators=[". ", "\n"],
    )
    docs = [Document(page_content=c) for c in splitter.split_text(text)]

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = InMemoryVectorStore(embeddings)
    store.add_documents(docs)
    return store


def build_chain(store: InMemoryVectorStore):
    retriever = store.as_retriever(search_kwargs={"k": 2})
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    def rag_answer(question: str) -> dict:
        docs = retriever.invoke(question)
        context = "\n".join(d.page_content for d in docs)
        msg = PROMPT.invoke({"context": context, "question": question})
        answer = llm.invoke(msg).content.strip()
        return {"answer": answer, "contexts": [d.page_content for d in docs]}

    return rag_answer


if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("Set GROQ_API_KEY (free key from https://console.groq.com) and re-run.")

    store = build_vector_store()
    chain = build_chain(store)

    for q in ASSIGNMENT_QUESTIONS:
        result = chain(q)
        print(f"Q: {q}")
        for c in result["contexts"]:
            print(f"   [retrieved] {c}")
        print(f"A: {result['answer']}\n")
