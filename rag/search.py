"""HybridSearcher: one object that ingests files and answers search queries."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .bm25_index import BM25Index
from .chunker import Chunk, chunk_documents
from .config import Settings
from .embeddings import Embedder, get_embedder
from .fusion import reciprocal_rank_fusion
from .loader import load_path
from .vector_store import VectorStore

Mode = Literal["hybrid", "bm25", "vector"]


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    bm25_rank: int | None  # 1-based rank in the keyword list, None if absent
    vector_rank: int | None  # 1-based rank in the vector list, None if absent

    def to_dict(self) -> dict:
        return {
            **self.chunk.to_dict(),
            "score": self.score,
            "bm25_rank": self.bm25_rank,
            "vector_rank": self.vector_rank,
        }


class HybridSearcher:
    def __init__(self, settings: Settings | None = None, embedder: Embedder | None = None):
        self.s = settings or Settings()
        self.s.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or get_embedder()
        self.vectors = VectorStore(self.s.index_dir / "chroma", self.s.collection, self.embedder)
        self._chunks_path = self.s.index_dir / "chunks.json"
        self.chunks: dict[str, Chunk] = self._load_chunks()
        self._rebuild_bm25()

    # ---------- persistence ----------
    def _load_chunks(self) -> dict[str, Chunk]:
        if not self._chunks_path.exists():
            return {}
        data = json.loads(self._chunks_path.read_text())
        return {d["id"]: Chunk(**d) for d in data}

    def _save_chunks(self) -> None:
        self._chunks_path.write_text(json.dumps([c.to_dict() for c in self.chunks.values()]))

    def _rebuild_bm25(self) -> None:
        ids = list(self.chunks)
        self.bm25 = BM25Index(ids, [self.chunks[i].text for i in ids])

    # ---------- ingest ----------
    def ingest(self, path: str | Path, reset: bool = False) -> int:
        """Load, chunk and index files. Returns the number of new chunks added."""
        if reset:
            self.vectors.reset()
            self.chunks = {}
        new = [
            c
            for c in chunk_documents(load_path(path), self.s.chunk_size, self.s.chunk_overlap)
            if c.id not in self.chunks
        ]
        if new:
            self.vectors.add(new)
            self.chunks.update({c.id: c for c in new})
        self._save_chunks()
        self._rebuild_bm25()
        return len(new)

    # ---------- search ----------
    def search(self, query: str, k: int = 5, mode: Mode = "hybrid") -> list[SearchResult]:
        n = max(k, self.s.candidate_k)
        bm25_hits = self.bm25.search(query, n) if mode in ("hybrid", "bm25") else []
        vec_hits = self.vectors.search(query, n) if mode in ("hybrid", "vector") else []

        bm25_rank = {cid: r for r, (cid, _) in enumerate(bm25_hits, start=1)}
        vec_rank = {cid: r for r, (cid, _) in enumerate(vec_hits, start=1)}

        if mode == "bm25":
            ranked = bm25_hits
        elif mode == "vector":
            ranked = vec_hits
        else:
            ranked = reciprocal_rank_fusion(
                [[c for c, _ in bm25_hits], [c for c, _ in vec_hits]], k=self.s.rrf_k
            )

        return [
            SearchResult(self.chunks[cid], score, bm25_rank.get(cid), vec_rank.get(cid))
            for cid, score in ranked[:k]
            if cid in self.chunks
        ]

    def stats(self) -> dict:
        return {
            "chunks": len(self.chunks),
            "vectors": self.vectors.count(),
            "sources": len({c.source for c in self.chunks.values()}),
            "embedder": self.embedder.name,
            "index_dir": str(self.s.index_dir),
        }
