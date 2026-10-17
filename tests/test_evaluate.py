"""Eval harness tests, with the offline hashing embedder (no model download)."""

import json
from pathlib import Path

import pytest

from rag.config import Settings
from rag.embeddings import HashingEmbedder
from rag.evaluate import Question, check_labels, load_questions, run, score_question, to_markdown
from rag.search import HybridSearcher

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample"
QUESTIONS = ROOT / "eval" / "questions.jsonl"


@pytest.fixture(scope="module")
def searcher(tmp_path_factory):
    hs = HybridSearcher(Settings(index_dir=tmp_path_factory.mktemp("idx")), embedder=HashingEmbedder())
    hs.ingest(SAMPLE, reset=True)
    return hs


def test_every_label_matches_a_chunk(searcher):
    qs = load_questions(QUESTIONS)
    assert len(qs) >= 25 and len({q.id for q in qs}) == len(qs)
    check_labels(searcher, qs)  # raises if any label points at nothing


def test_bad_label_is_an_error(searcher):
    q = Question("x", "keyword", "?", [{"source": "refunds.md", "contains": "no such phrase"}])
    with pytest.raises(ValueError, match="match no chunk"):
        check_labels(searcher, [q])


def test_metrics_for_an_exact_hit(searcher):
    q = Question("x", "keyword", "ERR-4220", [{"source": "refunds.md", "contains": "ERR-4220"}])
    row = score_question(searcher, q, "bm25")
    assert row["first_rank"] == 1 and row["mrr"] == 1.0 and row["recall@1"] == 1.0


def test_partial_recall_with_two_labels(searcher):
    q = Question("x", "keyword", "ERR-4220", [
        {"source": "refunds.md", "contains": "ERR-4220"},
        {"source": "webhooks.md", "contains": "HMAC-SHA256"},
    ])
    row = score_question(searcher, q, "bm25")
    assert row["recall@5"] == 0.5 and row["mrr"] == 1.0


def test_run_and_report(tmp_path):
    qs = tmp_path / "q.jsonl"
    qs.write_text("\n".join(QUESTIONS.read_text().splitlines()[:4]) + "\n")
    res = run(qs, SAMPLE, HashingEmbedder())
    assert res["questions"] == 4 and set(res["summary"]) == {"bm25", "vector", "hybrid"}
    for m in res["summary"].values():
        assert 0.0 <= m["recall@1"] <= m["recall@3"] <= m["recall@5"] <= 1.0
    json.dumps(res)
    assert "| hybrid |" in to_markdown(res)
