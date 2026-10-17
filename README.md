# RAG with hybrid search

Search your own documents by combining keyword search (BM25) with
meaning-based search (local embeddings in ChromaDB), merged with
Reciprocal Rank Fusion. Everything runs free and locally: no paid API is needed. This is part 1 of the project: the retrieval core.
LLM answers, the FastAPI app, evaluation and Docker come next.

## Quickstart

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python cli.py ingest data/sample --reset
python cli.py search "how do I rotate an API key?"
python cli.py search "ERR-4012" --mode bm25
python cli.py compare "what happens if a card is declined"   # bm25 vs vector vs hybrid
python cli.py stats
pytest -q
```

## Models (all free by default)

| Step | Default (free, local) | Optional |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers, on CPU | OpenAI `text-embedding-3-small` (paid): `RAG_EMBEDDINGS=openai` + `OPENAI_API_KEY` |
| Answers (part 2) | `llama3.2` via [Ollama](https://ollama.com) (`ollama run llama3.2`) | |

Pick the embedder with `RAG_EMBEDDINGS`: `minilm` (default), `openai`, or
`hashing`. The MiniLM model (~90 MB) downloads from Hugging Face on first run,
then works offline. If it can't load, the app prints a warning and falls back
to `hashing`, which only matches shared words and is meant for tests.

The index remembers which embedder built it; after switching, run
`ingest --reset`.

## How it works

```
files (.pdf .md .txt)
   -> loader.py         one Document per file (per page for PDFs)
   -> chunker.py        LangChain splitters, 600 chars, 100 overlap
   -> bm25_index.py     rank_bm25 over tokenized chunks   -> top 20 ids
   -> vector_store.py   MiniLM embeddings in Chroma       -> top 20 ids
   -> fusion.py         RRF: score = sum 1 / (60 + rank)
   -> top k chunks with source, page, and each retriever's rank
```

RRF uses ranks, not raw scores, so BM25 scores and cosine similarities never
need to be normalized against each other. A chunk that both retrievers rank
highly beats one that only a single retriever likes.

## Use from Python

```python
from rag import HybridSearcher

hs = HybridSearcher()
hs.ingest("path/to/docs")
for r in hs.search("refund window", k=5, mode="hybrid"):
    print(r.chunk.source, r.score, r.bm25_rank, r.vector_rank)
```

## Layout

| File | Role |
|---|---|
| `rag/loader.py` | Read PDF, Markdown and text files |
| `rag/chunker.py` | Split into overlapping chunks with stable ids |
| `rag/embeddings.py` | MiniLM (default), OpenAI (optional), hashing (tests) |
| `rag/vector_store.py` | Persistent ChromaDB collection (cosine) |
| `rag/bm25_index.py` | BM25 keyword index |
| `rag/fusion.py` | Reciprocal Rank Fusion |
| `rag/search.py` | `HybridSearcher`: ingest + search + stats |
| `cli.py` | Command line |
| `data/sample/` | Fictional "Nimbus Pay" API docs to try it on |

The index lives in `.index/` (Chroma data plus `chunks.json`, which BM25 is
rebuilt from on start). Re-ingesting the same files adds nothing new.
