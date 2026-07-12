# Custom RAG & Evaluation — Advanced Beekeeping Techniques

AI Engineer Intern Assignment (Level 1 Foundations). A Retrieval-Augmented
Generation (RAG) pipeline over a niche beekeeping dataset, with an evaluation
script that scores generated answers against expected answers using keyword
overlap and cosine similarity.

## What is RAG?

An LLM only knows what it saw during training. RAG fixes that by **retrieving**
relevant documents at question time and pasting them into the prompt, so the
model **generates** an answer grounded in real source text instead of guessing.

Pipeline:

```
Document ──> Chunking ──> Embeddings ──> Vector index
                                              │
Question ──> Embedding ──> similarity search ─┘──> top-k chunks
                                                        │
                              LLM prompt: context + question ──> Answer
```

## Project structure

| File | Purpose |
|---|---|
| `data/beekeeping.txt` | The provided dataset (Document 1) |
| `rag_pipeline.py` | RAG pipeline built from scratch (numpy + sentence-transformers + Groq) |
| `rag_langchain.py` | Same pipeline using the LangChain framework |
| `evaluate.py` | Scores answers vs. expected answers (keyword overlap + cosine similarity) |
| `requirements.txt` | Dependencies |

## Tech stack (100% free)

- **Embeddings:** `all-MiniLM-L6-v2` (sentence-transformers) — runs locally, no API key
- **Vector search:** cosine similarity via numpy (simple version) / `InMemoryVectorStore` (LangChain version)
- **LLM:** `llama-3.1-8b-instant` via the Groq free API
- **Fallback:** if `GROQ_API_KEY` is not set, `rag_pipeline.py` returns the top retrieved chunk (extractive mode), so everything still runs offline

## Setup

```bash
git clone <your-repo-url>
cd beekeeping-rag
python -m venv venv
venv\Scripts\activate        # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

Get a free API key at https://console.groq.com (no credit card), then:

```bash
set GROQ_API_KEY=your_key_here      # Windows CMD
# $env:GROQ_API_KEY="your_key"      # PowerShell
# export GROQ_API_KEY=your_key      # macOS/Linux
```

## Run

```bash
python rag_pipeline.py        # answer the 3 assignment questions (shows retrieved chunks + similarity scores)
python rag_langchain.py       # same, via LangChain
python evaluate.py            # evaluation: keyword overlap + cosine similarity per question
python evaluate.py --langchain
```

## Evaluation method

For each question the generated answer is compared to the expected answer:

1. **Keyword overlap** — expected answer is tokenized, stopwords removed; score = fraction of expected keywords found in the generated answer.
2. **Cosine similarity** — both answers embedded with `all-MiniLM-L6-v2`; score = cosine of the two vectors.

A question **passes** if keyword overlap ≥ 0.6 **or** cosine similarity ≥ 0.70.

### Sample output

```
Q:         Why are entrance reducers used?
Expected:  To prevent field mice from entering.
Generated: Entrance reducers are used to prevent field mice from entering during the colder months.
Keyword overlap: 1.00   Cosine similarity: 0.93   -> PASS

RESULT: 3/3 questions passed
```

## Design decisions

- **Sentence-level chunks:** the document is 4 sentences; one fact per chunk gives the retriever maximum precision. Larger corpora would use `RecursiveCharacterTextSplitter` with overlap (shown in the LangChain version).
- **top-k = 2:** enough context for any of the 3 questions without stuffing the prompt.
- **temperature = 0:** deterministic answers, essential for reproducible evaluation.
- **Grounded prompt:** the LLM is told to answer ONLY from context and say "I don't know" otherwise — this suppresses hallucination.

## Video walkthrough

Link: `<paste your video link here>`
