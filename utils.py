"""
utils.py — shared helpers for the RAG Bootcamp notebooks.

Keep this small and explicit. Consultants should be able to read the whole file
in 5 minutes and know exactly what's going on. No magic.
"""

from __future__ import annotations

import os
import json
import time
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

# Grok swap (uncomment if you prefer Grok) -------------------------
def get_llm_client():
    from openai import OpenAI   # Grok uses the OpenAI SDK
    client = OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )

    def generate(system: str, user: str,
                 model: str = "llama-3.1-8b-instant",
                 max_tokens: int = 1024) -> str:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content

    return client, generate

# ---------------------------------------------------------------------------
# Chunk dataclass — the unit of storage across the week
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single retrievable unit of text plus metadata."""
    id: str                             # stable hash of source + position
    text: str                           # the chunk content
    source: str                         # filename or URL
    page: int | None = None
    section: str | None = None          # e.g. "II.A.1" for nested structures
    content_type: str = "prose"         # prose | table | list | heading
    document_type: str | None = None    # contract | report | newsletter | etc
    extra: dict = field(default_factory=dict)

    def metadata(self) -> dict:
        """Chroma-compatible metadata (scalars only, no None)."""
        md = {
            "source": self.source,
            "content_type": self.content_type,
        }
        if self.page is not None:
            md["page"] = self.page
        if self.section:
            md["section"] = self.section
        if self.document_type:
            md["document_type"] = self.document_type
        for k, v in self.extra.items():
            if isinstance(v, (str, int, float, bool)):
                md[k] = v
        return md


def make_chunk_id(source: str, idx: int, text: str) -> str:
    """Deterministic ID so re-ingesting the same doc doesn't duplicate."""
    h = hashlib.md5(f"{source}|{idx}|{text[:80]}".encode()).hexdigest()[:16]
    return f"{Path(source).stem}_{idx}_{h}"


# ---------------------------------------------------------------------------
# Chroma helpers
# ---------------------------------------------------------------------------

def get_chroma_client(persist_dir: str = "./chroma_db"):
    import chromadb
    return chromadb.PersistentClient(path=persist_dir)


def add_chunks(collection, chunks: list[Chunk], embeddings: list[list[float]] | None = None):
    """Add a batch of chunks to a Chroma collection."""
    if not chunks:
        return
    ids = [c.id for c in chunks]
    docs = [c.text for c in chunks]
    mds = [c.metadata() for c in chunks]
    kwargs = {"ids": ids, "documents": docs, "metadatas": mds}
    if embeddings is not None:
        kwargs["embeddings"] = embeddings
    collection.upsert(**kwargs)


# ---------------------------------------------------------------------------
# Simple evaluation helpers
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 5) -> float:
    """Fraction of relevant IDs that appear in the top-k retrieved."""
    if not relevant_ids:
        return 0.0
    top = set(retrieved_ids[:k])
    hits = sum(1 for r in relevant_ids if r in top)
    return hits / len(relevant_ids)


def load_eval_set(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def save_eval_set(path: str, data: list[dict]):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def check_env() -> dict:
    """Return a dict of what's available. Useful in Day 0."""
    status = {}
    status["ANTHROPIC_API_KEY"] = bool(os.environ.get("ANTHROPIC_API_KEY"))
    status["OPENAI_API_KEY"] = bool(os.environ.get("OPENAI_API_KEY"))

    def try_import(name):
        try:
            __import__(name)
            return True
        except ImportError:
            return False

    for pkg in ["anthropic", "openai", "chromadb", "sentence_transformers",
                "pdfplumber", "pypdf", "pytesseract", "pdf2image", "cv2",
                "docling", "rank_bm25", "ragas", "bs4", "trafilatura", "requests"]:
        status[pkg] = try_import(pkg)
    return status


def pretty(d: dict):
    """Print a dict nicely."""
    for k, v in d.items():
        mark = "✓" if v else "✗"
        print(f"  {mark} {k}: {v}")
