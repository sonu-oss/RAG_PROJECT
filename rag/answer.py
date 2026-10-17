"""Answer a question from the top hybrid-search results, with numbered citations.

The retrieved chunks are shown to the LLM as [1], [2], ... and it is told to
cite them inline. We then parse which numbers it actually used, so the UI can
show only the sources the answer relies on.
"""

import re
from dataclasses import dataclass, field

from .llm import LLM
from .search import HybridSearcher, Mode, SearchResult

NO_ANSWER = "I couldn't find that in the documents."

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered sources the user provides.
Rules:
- Cite every claim with the source number in square brackets, like [1] or [2][3].
- Only use numbers that appear in the sources list.
- If the sources don't contain the answer, reply exactly: "{NO_ANSWER}"
- Be concise: a few sentences or a short list."""

_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class Citation:
    number: int  # the [n] used in the answer
    result: SearchResult

    def to_dict(self) -> dict:
        return {"number": self.number, **self.result.to_dict(), "retrievers": retrievers(self.result)}


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[Citation]  # sources the answer cites, in [n] order
    sources: list[Citation] = field(default_factory=list)  # everything the LLM was shown
    model: str = ""

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "sources": [c.to_dict() for c in self.sources],
            "model": self.model,
        }


def retrievers(r: SearchResult) -> list[str]:
    """Which retrievers ranked this chunk: ["bm25"], ["vector"] or both."""
    out = []
    if r.bm25_rank is not None:
        out.append("bm25")
    if r.vector_rank is not None:
        out.append("vector")
    return out


def build_prompt(question: str, results: list[SearchResult]) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        loc = r.chunk.source + (f", page {r.chunk.page}" if r.chunk.page else "")
        blocks.append(f"[{i}] ({loc})\n{r.chunk.text}")
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\nAnswer with citations:"


def cited_numbers(text: str, n_sources: int) -> list[int]:
    """Numbers cited as [n] or [n, m], in first-use order, ignoring out-of-range ones."""
    seen: list[int] = []
    for group in _CITE.findall(text):
        for part in group.split(","):
            n = int(part)
            if 1 <= n <= n_sources and n not in seen:
                seen.append(n)
    return seen


class Answerer:
    def __init__(self, searcher: HybridSearcher, llm: LLM):
        self.searcher = searcher
        self.llm = llm

    def ask(self, question: str, k: int = 5, mode: Mode = "hybrid") -> Answer:
        results = self.searcher.search(question, k=k, mode=mode)
        sources = [Citation(i, r) for i, r in enumerate(results, 1)]
        if not results:
            return Answer(question, NO_ANSWER, [], [], self.llm.name)

        text = self.llm.chat(SYSTEM_PROMPT, build_prompt(question, results))
        used = cited_numbers(text, len(results))
        return Answer(
            question=question,
            answer=text or NO_ANSWER,
            citations=[sources[n - 1] for n in used],
            sources=sources,
            model=self.llm.name,
        )
