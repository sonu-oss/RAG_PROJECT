from pathlib import Path

import pytest

from rag.config import Settings
from rag.embeddings import HashingEmbedder
from rag.fusion import reciprocal_rank_fusion
from rag.search import HybridSearcher

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample"


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], k=60)
    ids = [i for i, _ in fused]
    assert ids[0] == "b"  # ranked well by both lists
    assert set(ids) == {"a", "b", "c", "d"}


def test_rrf_weights():
    fused = reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0, 2.0])
    assert fused[0][0] == "b"


@pytest.fixture(scope="module")
def searcher(tmp_path_factory):
    s = Settings(index_dir=tmp_path_factory.mktemp("idx"))
    hs = HybridSearcher(settings=s, embedder=HashingEmbedder())
    assert hs.ingest(SAMPLE, reset=True) > 0
    return hs


@pytest.mark.parametrize("mode", ["bm25", "vector", "hybrid"])
def test_exact_error_code(searcher, mode):
    top = searcher.search("ERR-4012", k=3, mode=mode)
    assert top and top[0].chunk.source == "authentication.md"


def test_hybrid_reports_ranks(searcher):
    res = searcher.search("webhook signature verification", k=3)
    assert res[0].chunk.source == "webhooks.md"
    assert res[0].bm25_rank is not None or res[0].vector_rank is not None


def test_reingest_is_idempotent(searcher):
    before = searcher.stats()["chunks"]
    assert searcher.ingest(SAMPLE) == 0
    assert searcher.stats()["chunks"] == before == searcher.stats()["vectors"]


def test_index_reloads_from_disk(searcher):
    again = HybridSearcher(settings=searcher.s, embedder=HashingEmbedder())
    assert again.stats()["chunks"] == searcher.stats()["chunks"]
    assert again.search("refund window", k=1)[0].chunk.source == "refunds.md"
