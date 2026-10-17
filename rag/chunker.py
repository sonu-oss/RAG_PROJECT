"""Split documents into overlapping chunks with LangChain's recursive splitter."""

import hashlib
from dataclasses import dataclass, asdict

from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter

from .loader import Document


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int | None
    chunk_index: int

    def to_dict(self) -> dict:
        return asdict(self)


def _chunk_id(source: str, page: int | None, idx: int, text: str) -> str:
    raw = f"{source}|{page}|{idx}|{text}".encode()
    return hashlib.sha1(raw).hexdigest()[:16]


def chunk_documents(docs: list[Document], chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    md = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    plain = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    chunks: list[Chunk] = []
    for doc in docs:
        splitter = md if doc.source.lower().endswith((".md", ".markdown")) else plain
        for i, piece in enumerate(splitter.split_text(doc.text)):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    id=_chunk_id(doc.source, doc.page, i, piece),
                    text=piece,
                    source=doc.source,
                    page=doc.page,
                    chunk_index=i,
                )
            )
    return chunks
