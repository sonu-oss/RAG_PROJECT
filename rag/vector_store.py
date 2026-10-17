"""Dense retrieval with a persistent ChromaDB collection (cosine distance)."""

from pathlib import Path

import chromadb

from .chunker import Chunk
from .embeddings import Embedder


class VectorStore:
    def __init__(self, path: Path, collection: str, embedder: Embedder):
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection_name = collection
        self.col = self.client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine", "embedder": embedder.name}
        )
        stored = (self.col.metadata or {}).get("embedder")
        if stored and stored != embedder.name:
            raise RuntimeError(
                f"Index was built with '{stored}' but the current embedder is '{embedder.name}'. "
                "Re-run ingest with --reset, or switch embedders back."
            )

    def reset(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.col = self.client.get_or_create_collection(
            self.collection_name, metadata={"hnsw:space": "cosine", "embedder": self.embedder.name}
        )

    def add(self, chunks: list[Chunk], batch: int = 256) -> None:
        for i in range(0, len(chunks), batch):
            part = chunks[i : i + batch]
            self.col.upsert(
                ids=[c.id for c in part],
                documents=[c.text for c in part],
                embeddings=self.embedder.embed([c.text for c in part]),
                metadatas=[
                    {"source": c.source, "page": c.page or 0, "chunk_index": c.chunk_index}
                    for c in part
                ],
            )

    def count(self) -> int:
        return self.col.count()

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return (chunk_id, cosine similarity) best first."""
        n = self.count()
        if n == 0:
            return []
        res = self.col.query(
            query_embeddings=self.embedder.embed([query]), n_results=min(k, n), include=["distances"]
        )
        return [(cid, 1.0 - d) for cid, d in zip(res["ids"][0], res["distances"][0])]
