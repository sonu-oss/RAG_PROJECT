"""FastAPI backend.

    uvicorn api:app --reload

Then open http://localhost:8000/docs to try every endpoint in the browser.
"""

import shutil
import threading
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from rag import HybridSearcher
from rag.answer import Answerer
from rag.llm import LLM, LLMError, get_llm
from rag.loader import SUPPORTED

UPLOAD_DIR = Path("data/uploads")

app = FastAPI(title="RAG with hybrid search", version="0.2.0")
_ingest_lock = threading.Lock()


@lru_cache
def get_searcher() -> HybridSearcher:
    return HybridSearcher()


@lru_cache
def get_model() -> LLM:
    return get_llm()


class IngestRequest(BaseModel):
    path: str = Field("data/sample", description="File or folder on the server to index")
    reset: bool = Field(False, description="Drop the existing index first")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=50)
    mode: Literal["hybrid", "bm25", "vector"] = "hybrid"


class AskRequest(SearchRequest):
    pass


@app.get("/health")
def health(hs: HybridSearcher = Depends(get_searcher), llm: LLM = Depends(get_model)):
    return {"status": "ok", "llm": llm.name, **hs.stats()}


@app.post("/ingest")
def ingest(req: IngestRequest, hs: HybridSearcher = Depends(get_searcher)):
    if not Path(req.path).exists():
        raise HTTPException(404, f"Path not found on the server: {req.path}")
    with _ingest_lock:
        added = hs.ingest(req.path, reset=req.reset)
    return {"added": added, **hs.stats()}


@app.post("/ingest/upload")
def ingest_upload(
    files: list[UploadFile] = File(...), hs: HybridSearcher = Depends(get_searcher)
):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved, skipped = [], []
    for f in files:
        name = Path(f.filename or "").name  # drop any directory part
        if not name or Path(name).suffix.lower() not in SUPPORTED:
            skipped.append(f.filename)
            continue
        with open(UPLOAD_DIR / name, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(name)
    added = 0
    with _ingest_lock:
        for name in saved:
            try:
                added += hs.ingest(UPLOAD_DIR / name)
            except Exception as e:  # surface the reason in the UI instead of a bare 500
                raise HTTPException(500, f"Couldn't index {name}: {type(e).__name__}: {e}") from e
    return {"added": added, "saved": saved, "skipped": skipped, **hs.stats()}


@app.get("/sources")
def sources(hs: HybridSearcher = Depends(get_searcher)):
    """Every indexed file with its number of chunks."""
    counts: dict[str, int] = {}
    for c in hs.chunks.values():
        counts[c.source] = counts.get(c.source, 0) + 1
    return {"sources": [{"source": k, "chunks": v} for k, v in sorted(counts.items())]}


@app.post("/search")
def search(req: SearchRequest, hs: HybridSearcher = Depends(get_searcher)):
    return {"query": req.query, "mode": req.mode, "results": [r.to_dict() for r in hs.search(req.query, req.k, req.mode)]}


@app.post("/ask")
def ask(
    req: AskRequest,
    hs: HybridSearcher = Depends(get_searcher),
    llm: LLM = Depends(get_model),
):
    try:
        return Answerer(hs, llm).ask(req.query, k=req.k, mode=req.mode).to_dict()
    except LLMError as e:
        raise HTTPException(503, str(e)) from e
