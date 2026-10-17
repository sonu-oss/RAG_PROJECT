"""Keyword retrieval with rank_bm25 (Okapi BM25)."""

import re

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_STOP = set(
    "a an and are as at be by for from has have how i in is it its of on or that the this to was "
    "what when where which who why will with you your do does did can".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25Index:
    def __init__(self, ids: list[str], texts: list[str]):
        self.ids = ids
        self._bm25 = BM25Okapi([tokenize(t) for t in texts]) if texts else None

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return (chunk_id, score) best first. Chunks with no term overlap are dropped."""
        if self._bm25 is None:
            return []
        q = tokenize(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(self.ids[i], float(scores[i])) for i in ranked[:k] if scores[i] > 0]
