"""Hybrid (BM25 + vector) retrieval for a chat-with-your-docs RAG app."""

from .search import HybridSearcher, SearchResult

__all__ = ["HybridSearcher", "SearchResult"]
