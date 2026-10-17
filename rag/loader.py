"""Load .pdf, .md and .txt files into plain-text documents."""

from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".pdf", ".md", ".markdown", ".txt"}


@dataclass
class Document:
    text: str
    source: str  # file path relative to the ingest root
    page: int | None = None  # 1-based page number for PDFs


def _load_pdf(path: Path, source: str) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    docs = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(text=text, source=source, page=i))
    return docs


def load_path(root: str | Path) -> list[Document]:
    """Load one file, or every supported file under a directory (recursive)."""
    root = Path(root)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    base = root.parent if root.is_file() else root

    docs: list[Document] = []
    for path in files:
        ext = path.suffix.lower()
        if ext not in SUPPORTED:
            continue
        source = str(path.relative_to(base))
        if ext == ".pdf":
            docs.extend(_load_pdf(path, source))
        else:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                docs.append(Document(text=text, source=source))
    return docs
