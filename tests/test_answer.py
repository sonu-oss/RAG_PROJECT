"""Answer layer and API tests. Ollama is never called: the LLM is faked."""

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import api
from rag.answer import NO_ANSWER, Answerer, build_prompt, cited_numbers
from rag.config import Settings
from rag.embeddings import HashingEmbedder
from rag.llm import LLMError, OllamaLLM
from rag.search import HybridSearcher

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample"


class FakeLLM:
    name = "fake:test"

    def __init__(self, reply="Rotate keys in the dashboard [1]. Old keys stop working [2][9]."):
        self.reply = reply
        self.calls = []

    def chat(self, system, user):
        self.calls.append((system, user))
        return self.reply


@pytest.fixture(scope="module")
def searcher(tmp_path_factory):
    hs = HybridSearcher(Settings(index_dir=tmp_path_factory.mktemp("idx")), embedder=HashingEmbedder())
    hs.ingest(SAMPLE, reset=True)
    return hs


def test_cited_numbers_order_dedupe_and_range():
    assert cited_numbers("a [2] b [1, 2] c [7] d [3][2]", n_sources=3) == [2, 1, 3]
    assert cited_numbers("no citations here", 5) == []


def test_prompt_numbers_sources(searcher):
    results = searcher.search("rotate api key", k=3)
    prompt = build_prompt("How?", results)
    for i, r in enumerate(results, 1):
        assert f"[{i}] ({r.chunk.source}" in prompt
    assert prompt.rstrip().endswith("Answer with citations:")


def test_ask_maps_citations_to_chunks(searcher):
    llm = FakeLLM()
    ans = Answerer(searcher, llm).ask("how do I rotate an API key?", k=3)
    assert len(llm.calls) == 1
    assert [c.number for c in ans.citations] == [1, 2]  # [9] is out of range, dropped
    assert len(ans.sources) == 3
    assert ans.citations[0].result is ans.sources[0].result
    d = ans.to_dict()
    assert d["model"] == "fake:test"
    assert set(d["citations"][0]["retrievers"]) <= {"bm25", "vector"}
    assert d["citations"][0]["retrievers"]


def test_ask_with_empty_index_skips_llm(tmp_path):
    empty = HybridSearcher(Settings(index_dir=tmp_path), embedder=HashingEmbedder())
    llm = FakeLLM()
    ans = Answerer(empty, llm).ask("anything")
    assert ans.answer == NO_ANSWER and not ans.citations and not llm.calls


# ---------- Ollama client (HTTP mocked) ----------
def test_ollama_parses_reply(monkeypatch):
    seen = {}

    def fake_post(url, json, timeout):
        seen.update(url=url, body=json)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": " Hi [1] "}})

    monkeypatch.setattr(httpx, "post", fake_post)
    out = OllamaLLM(model="llama3.2", base_url="http://ollama:11434/").chat("sys", "user")
    assert out == "Hi [1]"
    assert seen["url"] == "http://ollama:11434/api/chat"
    assert seen["body"]["model"] == "llama3.2" and seen["body"]["stream"] is False


def test_ollama_down_gives_helpful_error(monkeypatch):
    def fake_post(*a, **kw):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(LLMError, match="ollama serve"):
        OllamaLLM().chat("s", "u")


def test_ollama_missing_model(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: httpx.Response(404, text="model not found"))
    with pytest.raises(LLMError, match="ollama pull llama3.2"):
        OllamaLLM(model="llama3.2").chat("s", "u")


# ---------- FastAPI ----------
@pytest.fixture
def client(tmp_path, monkeypatch):
    hs = HybridSearcher(Settings(index_dir=tmp_path / "idx"), embedder=HashingEmbedder())
    llm = FakeLLM("Refunds take 5-10 business days [1].")
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path / "uploads")
    api.app.dependency_overrides[api.get_searcher] = lambda: hs
    api.app.dependency_overrides[api.get_model] = lambda: llm
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def test_api_ingest_search_ask(client):
    assert client.get("/health").json()["chunks"] == 0
    r = client.post("/ingest", json={"path": str(SAMPLE), "reset": True})
    assert r.status_code == 200 and r.json()["added"] > 0

    r = client.post("/search", json={"query": "ERR-4012", "k": 3, "mode": "bm25"})
    assert r.json()["results"][0]["source"] == "authentication.md"

    r = client.post("/ask", json={"query": "how long do refunds take?", "k": 3})
    body = r.json()
    assert r.status_code == 200
    assert body["answer"].startswith("Refunds") and body["model"] == "fake:test"
    assert [c["number"] for c in body["citations"]] == [1]
    assert {"bm25_rank", "vector_rank", "retrievers", "text", "source"} <= body["citations"][0].keys()


def test_api_ingest_missing_path(client):
    assert client.post("/ingest", json={"path": "/no/such/dir"}).status_code == 404


def test_api_upload(client):
    files = [
        ("files", ("notes.md", b"# Notes\nThe secret code word is PINEAPPLE-42.")),
        ("files", ("image.png", b"\x89PNG")),
    ]
    r = client.post("/ingest/upload", files=files)
    assert r.json()["saved"] == ["notes.md"] and r.json()["skipped"] == ["image.png"]
    top = client.post("/search", json={"query": "PINEAPPLE-42", "k": 1}).json()["results"]
    assert top[0]["source"] == "notes.md"


def test_api_llm_down_returns_503(client):
    class DownLLM(FakeLLM):
        def chat(self, system, user):
            raise LLMError("Can't reach Ollama")

    client.post("/ingest", json={"path": str(SAMPLE)})
    api.app.dependency_overrides[api.get_model] = lambda: DownLLM()
    r = client.post("/ask", json={"query": "refunds"})
    assert r.status_code == 503 and "Ollama" in r.json()["detail"]


def test_api_sources_lists_files(client):
    client.post("/ingest", json={"path": str(SAMPLE)})
    client.post("/ingest/upload", files=[("files", ("prd.md", b"# PRD\nMust support SSO."))])
    names = [s["source"] for s in client.get("/sources").json()["sources"]]
    assert "prd.md" in names and "refunds.md" in names


def test_api_upload_index_error_is_readable(client, monkeypatch):
    def boom(*a, **kw):
        raise ValueError("bad file")

    hs = api.app.dependency_overrides[api.get_searcher]()
    monkeypatch.setattr(hs, "ingest", boom)
    r = client.post("/ingest/upload", files=[("files", ("x.md", b"hello"))])
    assert r.status_code == 500 and "x.md" in r.json()["detail"] and "bad file" in r.json()["detail"]
