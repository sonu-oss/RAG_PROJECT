"""Streamlit web UI. It talks to the FastAPI backend, so start that first:

    uvicorn api:app --reload        # terminal 1
    streamlit run ui.py             # terminal 2
"""

import os

import httpx
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Ask your docs", page_icon="🔎", layout="wide")


class APIError(Exception):
    pass


def try_api(method: str, path: str, **kw):
    """Call the backend; raise APIError with a readable message on failure."""
    try:
        r = httpx.request(method, f"{API_URL}{path}", timeout=300, **kw)
    except httpx.ConnectError as e:
        raise APIError(f"Can't reach the API at {API_URL}. Start it with `uvicorn api:app --reload`.") from e
    except httpx.TimeoutException as e:
        raise APIError("The API took too long to respond. Check the uvicorn terminal for errors.") from e
    if r.status_code >= 400:
        is_json = r.headers.get("content-type", "").startswith("application/json")
        detail = r.json().get("detail", r.text) if is_json else r.text
        raise APIError(f"API error {r.status_code}: {detail}")
    return r.json()


def api(method: str, path: str, **kw):
    """Like try_api, but show the error on the page and stop."""
    try:
        return try_api(method, path, **kw)
    except APIError as e:
        st.error(str(e))
        st.stop()


def flash(kind: str, msg: str):
    """Keep a message on screen across reruns, until the next one replaces it."""
    st.session_state["flash"] = (kind, msg)


def index_new_uploads(uploads) -> None:
    """Index any uploaded file we haven't sent to the API yet (no button needed)."""
    done: set = st.session_state.setdefault("indexed_uploads", set())
    new = [f for f in uploads or [] if (f.name, f.size) not in done]
    if not new:
        return
    with st.spinner(f"Indexing {', '.join(f.name for f in new)}..."):
        try:
            res = try_api("POST", "/ingest/upload", files=[("files", (f.name, f.getvalue())) for f in new])
        except APIError as e:
            flash("error", str(e))
            return
    done.update((f.name, f.size) for f in new)
    msg = f"Indexed {', '.join(res['saved']) or 'nothing'}: {res['added']} new chunks."
    if res["skipped"]:
        msg += f" Skipped (unsupported type): {', '.join(map(str, res['skipped']))}."
    if res["saved"] and res["added"] == 0:
        msg += " (Already indexed, nothing new.)"
    flash("success", msg)
    st.rerun()


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

    uploads = st.file_uploader(
        "Add your own files (indexed automatically)",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
    )
    index_new_uploads(uploads)

    if "flash" in st.session_state:
        kind, msg = st.session_state["flash"]
        (st.success if kind == "success" else st.error)(msg)

    with st.expander(f"Indexed files ({health['sources']})"):
        for s in api("GET", "/sources")["sources"]:
            st.markdown(f"- `{s['source']}` · {s['chunks']} chunks")

    if st.button("Index sample docs"):
        res = api("POST", "/ingest", json={"path": "data/sample", "reset": False})
        flash("success", f"Sample docs: {res['added']} new chunks.")
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
