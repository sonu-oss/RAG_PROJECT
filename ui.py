"""Streamlit web UI. It talks to the FastAPI backend, so start that first:

    uvicorn api:app --reload        # terminal 1
    streamlit run ui.py             # terminal 2
"""

import os

import httpx
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Ask your docs", page_icon="🔎", layout="wide")


def api(method: str, path: str, **kw):
    try:
        r = httpx.request(method, f"{API_URL}{path}", timeout=180, **kw)
    except httpx.ConnectError:
        st.error(f"Can't reach the API at {API_URL}. Start it with `uvicorn api:app --reload`.")
        st.stop()
    if r.status_code >= 400:
        detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
        st.error(f"{r.status_code}: {detail}")
        st.stop()
    return r.json()


def retriever_badges(src: dict) -> str:
    parts = []
    parts.append(f"BM25 #{src['bm25_rank']}" if src["bm25_rank"] else "BM25 –")
    parts.append(f"Vector #{src['vector_rank']}" if src["vector_rank"] else "Vector –")
    found = src["retrievers"]
    label = "both" if len(found) == 2 else (found[0] + " only" if found else "none")
    return f"`{parts[0]}` `{parts[1]}` · found by **{label}**"


def show_source(src: dict, cited: bool):
    loc = src["source"] + (f", p.{src['page']}" if src.get("page") else "")
    title = f"[{src['number']}] {loc}" + ("" if cited else "  (not cited)")
    with st.expander(title, expanded=cited):
        st.markdown(retriever_badges(src) + f" · RRF score `{src['score']:.4f}`")
        st.text(src["text"])


# ---------- sidebar: index ----------
with st.sidebar:
    st.header("Index")
    health = api("GET", "/health")
    st.caption(f"{health['chunks']} chunks from {health['sources']} files")
    st.caption(f"Embeddings: `{health['embedder']}`  \nLLM: `{health['llm']}`")

    if st.button("Index sample docs"):
        res = api("POST", "/ingest", json={"path": "data/sample", "reset": False})
        st.success(f"Added {res['added']} chunks")
        st.rerun()

    uploads = st.file_uploader("Add your own files", type=["pdf", "md", "txt"], accept_multiple_files=True)
    if uploads and st.button("Index uploaded files"):
        files = [("files", (f.name, f.getvalue())) for f in uploads]
        res = api("POST", "/ingest/upload", files=files)
        st.success(f"Added {res['added']} chunks from {len(res['saved'])} files")
        st.rerun()

    st.header("Retrieval")
    mode = st.radio("Mode", ["hybrid", "bm25", "vector"], horizontal=True)
    k = st.slider("Sources to use", 1, 10, 5)

# ---------- main: ask ----------
st.title("🔎 Ask your docs")
st.caption("Hybrid search (BM25 + vectors, fused with RRF) feeding a local llama3.2 through Ollama.")

question = st.text_input("Question", placeholder="How do I rotate an API key?")
if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Searching and asking the model..."):
        st.session_state["last"] = api("POST", "/ask", json={"query": question, "k": k, "mode": mode})

res = st.session_state.get("last")
if res:
    st.subheader("Answer")
    st.markdown(res["answer"])
    st.caption(f"Model: {res['model']}")

    cited = {c["number"] for c in res["citations"]}
    st.subheader("Sources")
    if not res["sources"]:
        st.info("No matching chunks. Try indexing some documents first.")
    for src in res["sources"]:
        show_source(src, src["number"] in cited)
