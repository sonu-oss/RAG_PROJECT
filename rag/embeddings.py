"""Embedding backends, picked with the RAG_EMBEDDINGS environment variable.

- "minilm" (default, free): sentence-transformers all-MiniLM-L6-v2, runs locally
  on CPU. The model (~90 MB) downloads from Hugging Face on first use.
- "openai" (optional, paid): text-embedding-3-small, needs OPENAI_API_KEY.
- "hashing" (offline, for tests): hashed word overlap, no download. It does not
  understand meaning, so don't use it for quality numbers.

If the MiniLM model can't be loaded (package missing, no network on first run),
the hashing embedder is used and a warning is printed.
"""

import hashlib
import math
import os
import re
import sys
from typing import Protocol

_TOKEN = re.compile(r"[a-z0-9]+")
DEFAULT_LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model: str = DEFAULT_LOCAL_MODEL, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model, device="cpu")
        self.batch_size = batch_size
        self.name = f"st:{model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return vecs.tolist()


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 128):
        from openai import OpenAI

        self.client = OpenAI()  # reads OPENAI_API_KEY from the environment
        self.model = model
        self.batch_size = batch_size
        self.name = f"openai:{model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t.replace("\n", " ") for t in texts[i : i + self.batch_size]]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            out.extend(d.embedding for d in resp.data)
        return out


class HashingEmbedder:
    """Offline fallback: hashed unigrams + bigrams, L2-normalised."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"local:hashing-{dim}"

    def _vec(self, text: str) -> list[float]:
        toks = _TOKEN.findall(text.lower())
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = [0.0] * self.dim
        for f in feats:
            h = int.from_bytes(hashlib.md5(f.encode()).digest()[:8], "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


def get_embedder(choice: str | None = None) -> Embedder:
    choice = (choice or os.getenv("RAG_EMBEDDINGS") or "minilm").lower()
    if choice == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("RAG_EMBEDDINGS=openai needs OPENAI_API_KEY to be set.")
        return OpenAIEmbedder(os.getenv("RAG_OPENAI_MODEL", "text-embedding-3-small"))
    if choice == "hashing":
        return HashingEmbedder()
    if choice == "minilm":
        try:
            return SentenceTransformerEmbedder(os.getenv("RAG_LOCAL_MODEL", DEFAULT_LOCAL_MODEL))
        except Exception as e:  # missing package, or model not downloadable
            print(
                f"[rag] Could not load the local embedding model ({type(e).__name__}: {e}). "
                "Falling back to the offline hashing embedder.",
                file=sys.stderr,
            )
            return HashingEmbedder()
    raise ValueError(f"Unknown RAG_EMBEDDINGS={choice!r}; use minilm, openai or hashing.")
