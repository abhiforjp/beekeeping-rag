"""
Evaluation script for the RAG pipeline.

For each question it compares the RAG-generated answer against the expected
answer using TWO metrics (the assignment asks for either; we do both):

  1. Keyword overlap  — fraction of the expected answer's keywords
                        (stopwords removed) that appear in the generated answer.
  2. Cosine similarity — semantic similarity between the embeddings of the
                        generated and expected answers (all-MiniLM-L6-v2).

A question PASSES if keyword overlap >= 0.6 OR cosine similarity >= 0.70.



import re
import sys

from sentence_transformers import SentenceTransformer

EVAL_SET = [
    {
        "question": "What is the minimum internal temperature for a Langstroth Hive in winter?",
        "expected": "Above 40 degrees Fahrenheit.",
    },
    {
        "question": "Why are entrance reducers used?",
        "expected": "To prevent field mice from entering.",
    },
    {
        "question": "How do beekeepers control condensation?",
        "expected": "By using insulated wraps and moisture quilt boxes.",
    },
]

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of",
    "in", "on", "at", "by", "for", "from", "with", "and", "or", "it", "its",
    "this", "that", "as", "do", "does", "how", "what", "why", "above",
}

KEYWORD_THRESHOLD = 0.6
COSINE_THRESHOLD = 0.70


def tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS}


def keyword_overlap(generated: str, expected: str) -> float:
    """Recall of expected-answer keywords inside the generated answer."""
    exp, gen = tokenize(expected), tokenize(generated)
    if not exp:
        return 0.0
    return len(exp & gen) / len(exp)


def main():
    use_langchain = "--langchain" in sys.argv

    # ---- get answers from the chosen pipeline -----------------------------
    if use_langchain:
        import rag_langchain as rl
        chain = rl.build_chain(rl.build_vector_store())
        get_answer = lambda q: chain(q)["answer"]
    else:
        import rag_pipeline as rp
        retriever = rp.build_retriever()
        get_answer = lambda q: rp.answer(q, retriever)[0]

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"\n{'=' * 78}")
    print(f"RAG EVALUATION ({'LangChain' if use_langchain else 'simple'} pipeline)")
    print(f"{'=' * 78}")

    passed = 0
    for item in EVAL_SET:
        q, expected = item["question"], item["expected"]
        generated = get_answer(q)

        overlap = keyword_overlap(generated, expected)
        embs = embedder.encode([generated, expected], normalize_embeddings=True)
        cosine = float(embs[0] @ embs[1])

        ok = overlap >= KEYWORD_THRESHOLD or cosine >= COSINE_THRESHOLD
        passed += ok

        print(f"\nQ:         {q}")
        print(f"Expected:  {expected}")
        print(f"Generated: {generated}")
        print(f"Keyword overlap: {overlap:.2f}   Cosine similarity: {cosine:.2f}"
              f"   -> {'PASS' if ok else 'FAIL'}")

    print(f"\n{'-' * 78}")
    print(f"RESULT: {passed}/{len(EVAL_SET)} questions passed "
          f"(keyword >= {KEYWORD_THRESHOLD} or cosine >= {COSINE_THRESHOLD})")
    print(f"{'-' * 78}\n")


if __name__ == "__main__":
    main()
