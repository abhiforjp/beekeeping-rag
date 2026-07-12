
import os
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "beekeeping.txt")
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 2  # how many chunks to retrieve per question

ASSIGNMENT_QUESTIONS = [
    "What is the minimum internal temperature for a Langstroth Hive in winter?",
    "Why are entrance reducers used?",
    "How do beekeepers control condensation?",
]


# ---------------------------------------------------------------- ingestion
def load_and_chunk(path: str = DATA_PATH) -> list[str]:
    """Read the dataset and split it into sentence-level chunks.

    The document is only 4 sentences, so one sentence per chunk gives the
    retriever maximum precision. For bigger corpora you would use a
    recursive/overlapping splitter instead.
    """
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()
    return [s.strip() + "." for s in text.split(".") if s.strip()]


# ---------------------------------------------------------------- retrieval
class Retriever:
    """Embeds chunks once, then answers queries with cosine similarity."""

    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.model = SentenceTransformer(EMBED_MODEL)
        # normalize_embeddings=True -> dot product == cosine similarity
        self.embeddings = self.model.encode(chunks, normalize_embeddings=True)

    def retrieve(self, query: str, k: int = TOP_K) -> list[tuple[str, float]]:
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ q_emb
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


# ---------------------------------------------------------------- generation
PROMPT_TEMPLATE = """You are a precise assistant. Answer the question using ONLY the context below.
Answer in one short sentence. If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""


def generate_answer(question: str, contexts: list[tuple[str, float]]) -> str:
    context_text = "\n".join(chunk for chunk, _ in contexts)
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        # Extractive fallback: return the single most relevant chunk.
        return contexts[0][0]

    from groq import Groq

    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": PROMPT_TEMPLATE.format(context=context_text, question=question),
        }],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------- pipeline
def answer(question: str, retriever: Retriever) -> tuple[str, list[tuple[str, float]]]:
    contexts = retriever.retrieve(question)
    return generate_answer(question, contexts), contexts


def build_retriever() -> Retriever:
    return Retriever(load_and_chunk())


if __name__ == "__main__":
    retriever = build_retriever()
    mode = "LLM (Groq)" if os.getenv("GROQ_API_KEY") else "extractive fallback (no GROQ_API_KEY set)"
    print(f"Generation mode: {mode}\n")

    questions = sys.argv[1:] or ASSIGNMENT_QUESTIONS
    for q in questions:
        ans, ctx = answer(q, retriever)
        print(f"Q: {q}")
        for chunk, score in ctx:
            print(f"   [retrieved | sim={score:.3f}] {chunk}")
        print(f"A: {ans}\n")
