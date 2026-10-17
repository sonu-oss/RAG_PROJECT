# RAG with hybrid search

Search your own documents by combining keyword search (BM25) with
meaning-based search (local embeddings in ChromaDB), merged with
Reciprocal Rank Fusion, then answer questions with a local LLM (llama3.2 via
Ollama) that cites its sources. Everything runs free and locally: no paid API
is needed. Done so far: the retrieval core (part 1) and LLM answers with a
FastAPI backend and Streamlit UI (part 2). Evaluation and Docker come next.

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

## Ask questions (LLM answers with citations)

Install [Ollama](https://ollama.com), open it once so it runs in the
background, then download the model (~2 GB, one time):

```bash
ollama pull llama3.2
python cli.py ask "how long do refunds take?"
```

### Web app

Run the API and the UI in two terminals (both with `.venv` activated):

```bash
uvicorn api:app --reload      # terminal 1: API on http://localhost:8000 (docs at /docs)
streamlit run ui.py           # terminal 2: UI on http://localhost:8501
```

The UI shows the answer, the numbered sources it cites, and for each source
its rank in BM25 and in vector search, so you can see which retriever found it.
You can switch between hybrid, BM25-only and vector-only, and upload your own
PDF, Markdown or text files.

| Endpoint | What it does |
|---|---|
| `GET /health` | Index stats and which embedder and LLM are in use |
| `POST /ingest` | Index a folder on the server: `{"path": "data/sample", "reset": false}` |
| `POST /ingest/upload` | Upload `.pdf` `.md` `.txt` files (multipart, field `files`) |
| `POST /search` | `{"query": "...", "k": 5, "mode": "hybrid"}` → ranked chunks |
| `POST /ask` | Same body → `answer`, `citations` (sources it cited), `sources` (all it was shown) |

If Ollama isn't running, `/ask` returns HTTP 503 with a message saying how to
start it. Use another model with `OLLAMA_MODEL=...`, or another host with
`OLLAMA_URL=...`.

## Models (all free by default)

| Step | Default (free, local) | Optional |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers, on CPU | OpenAI `text-embedding-3-small` (paid): `RAG_EMBEDDINGS=openai` + `OPENAI_API_KEY` |
| Answers | `llama3.2` via [Ollama](https://ollama.com) (`ollama pull llama3.2`) | any Ollama model: `OLLAMA_MODEL=...` |

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
   -> answer.py         numbered sources [1]..[k] -> llama3.2 (Ollama) -> answer with [n] citations
```

The prompt tells the model to answer only from the numbered sources, cite
each claim as `[n]`, and say "I couldn't find that in the documents." when the
sources don't cover it. Citations are parsed back out of the answer, and any
number outside the source list is dropped. With an empty index the LLM is not
called at all.

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
| `rag/llm.py` | Ollama client (local, free) |
| `rag/answer.py` | Prompt with numbered sources, parse `[n]` citations |
| `api.py` | FastAPI backend: `/ingest`, `/search`, `/ask` |
| `ui.py` | Streamlit UI on top of the API |
| `cli.py` | Command line |
| `data/sample/` | Fictional "Nimbus Pay" API docs to try it on |

The index lives in `.index/` (Chroma data plus `chunks.json`, which BM25 is
rebuilt from on start). Re-ingesting the same files adds nothing new.
