import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    index_dir: Path = field(default_factory=lambda: Path(os.getenv("RAG_INDEX_DIR", ".index")))
    collection: str = "docs"
    chunk_size: int = 600  # characters
    chunk_overlap: int = 100
    # Candidates pulled from each retriever before fusion.
    candidate_k: int = 20
    # RRF constant from Cormack et al. (2009); 60 is the usual default.
    rrf_k: int = 60
