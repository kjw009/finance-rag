from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import Chunk, get_chroma_client, add_chunks

EMBED_MODEL   = "all-MiniLM-L6-v2"
RERANK_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION    = "finance_rag"
BATCH_SIZE    = 64

TOP_K_DENSE  = 100
TOP_K_BM25   = 100
TOP_RERANK   = 50
TOP_K_FINAL  = 15
RRF_K        = 60

# Maps source filename prefixes to a canonical ticker.
# Order matters: more specific patterns first.
_SOURCE_PATTERNS: dict[str, list[str]] = {
    "META":  ["META_", "Meta-"],
    "AAPL":  ["AAPL_"],
    "GOOGL": ["GOOGL_", "goog-"],
    "AMZN":  ["AMZN_", "Amazon-"],
    "MSFT":  ["MSFT_"],
    "NVDA":  ["NVDA_", "NVIDIA-"],
    "TSLA":  ["TSLA_"],
    "AMD":   ["AMD_"],
}

# Maps query keywords (lowercase, exact word match) to ticker.
_QUERY_TICKER: list[tuple[set[str], str]] = [
    ({"meta", "facebook"},           "META"),
    ({"apple", "aapl"},              "AAPL"),
    ({"google", "alphabet", "googl"},"GOOGL"),
    ({"amazon", "amzn"},             "AMZN"),
    ({"microsoft", "msft"},          "MSFT"),
    ({"nvidia", "nvda"},             "NVDA"),
    ({"tesla", "tsla"},              "TSLA"),
    ({"amd"},                        "AMD"),
]


def _detect_ticker(query: str) -> str | None:
    import re
    words = set(re.sub(r"[^a-z0-9 ]", " ", query.lower()).split())
    for keywords, ticker in _QUERY_TICKER:
        if keywords & words:
            return ticker
    return None


def _source_matches(source: str, ticker: str) -> bool:
    return any(source.startswith(p) for p in _SOURCE_PATTERNS.get(ticker, []))


class Retriever:
    def __init__(
        self,
        all_chunks: list[Chunk],
        collection,
        bi_encoder: SentenceTransformer,
        bm25: BM25Okapi,
        cross_encoder: CrossEncoder,
    ):
        self.all_chunks    = all_chunks
        self.collection    = collection
        self.bi_encoder    = bi_encoder
        self.bm25          = bm25
        self.cross_encoder = cross_encoder
        self.id_to_idx     = {c.id: i for i, c in enumerate(all_chunks)}

        # Precompute per-ticker index sets and source lists for fast filtering.
        self._ticker_idxs: dict[str, set[int]] = {}
        self._ticker_sources: dict[str, list[str]] = {}
        for ticker in _SOURCE_PATTERNS:
            idxs = {i for i, c in enumerate(all_chunks) if _source_matches(c.source, ticker)}
            srcs = list({all_chunks[i].source for i in idxs})
            self._ticker_idxs[ticker]   = idxs
            self._ticker_sources[ticker] = srcs

    # ── Public entry point ────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K_FINAL) -> list[Chunk]:
        ticker     = _detect_ticker(query)
        valid_idxs = self._ticker_idxs.get(ticker) if ticker else None
        sources    = self._ticker_sources.get(ticker) if ticker else None

        dense  = self._dense(query, sources=sources)
        sparse = self._bm25(query, valid_idxs=valid_idxs)
        fused  = self._rrf(dense, sparse)
        ranked = self._rerank(query, fused, top_n=top_k)
        return [self.all_chunks[idx] for idx, _ in ranked]

    # ── Internal stages ───────────────────────────────────────────────────────

    def _dense(
        self,
        query: str,
        k: int = TOP_K_DENSE,
        sources: list[str] | None = None,
    ) -> list[tuple[int, float]]:
        vec    = self.bi_encoder.encode(query, convert_to_numpy=True).tolist()
        kwargs: dict = {"query_embeddings": [vec], "n_results": k, "include": ["distances"]}
        if sources:
            kwargs["where"] = {"source": {"$in": sources}}
        results = self.collection.query(**kwargs)
        hits = []
        for cid, dist in zip(results["ids"][0], results["distances"][0]):
            idx = self.id_to_idx.get(cid)
            if idx is not None:
                hits.append((idx, 1 / (1 + dist)))
        return hits

    def _bm25(
        self,
        query: str,
        k: int = TOP_K_BM25,
        valid_idxs: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(query.lower().split())
        order  = scores.argsort()[::-1]
        results = []
        for i in order:
            if valid_idxs is None or int(i) in valid_idxs:
                results.append((int(i), float(scores[i])))
                if len(results) >= k:
                    break
        return results

    def _rrf(
        self,
        *ranked_lists: list[tuple[int, float]],
        top_n: int = TOP_RERANK,
    ) -> list[int]:
        fused: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, (idx, _) in enumerate(ranked):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        return sorted(fused, key=fused.__getitem__, reverse=True)[:top_n]

    def _rerank(
        self, query: str, candidate_ids: list[int], top_n: int = TOP_K_FINAL
    ) -> list[tuple[int, float]]:
        pairs  = [(query, self.all_chunks[i].text) for i in candidate_ids]
        scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        return sorted(
            zip(candidate_ids, scores.tolist()), key=lambda x: x[1], reverse=True
        )[:top_n]

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_index(
        cls,
        chroma_dir: str,
        collection_name: str,
        all_chunks: list[Chunk],
    ) -> "Retriever":
        print("Loading bi-encoder …")
        bi_encoder = SentenceTransformer(EMBED_MODEL)
        print("Loading cross-encoder …")
        cross_encoder = CrossEncoder(RERANK_MODEL)
        print("Building BM25 index …")
        bm25 = BM25Okapi([c.text.lower().split() for c in all_chunks])
        collection = get_chroma_client(chroma_dir).get_collection(collection_name)
        print(f"Connected to collection '{collection_name}' ({collection.count()} docs)")
        return cls(all_chunks, collection, bi_encoder, bm25, cross_encoder)


def build_index(
    chunks: list[Chunk],
    chroma_dir: str,
    collection_name: str,
) -> None:
    """Embed all chunks and upsert into ChromaDB."""
    print(f"Loading embedding model '{EMBED_MODEL}' …")
    model      = SentenceTransformer(EMBED_MODEL)
    client     = get_chroma_client(chroma_dir)
    collection = client.get_or_create_collection(collection_name)

    texts = [c.text for c in chunks]
    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        vecs  = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.extend(vecs.tolist())
        done = min(start + BATCH_SIZE, len(texts))
        if (start // BATCH_SIZE) % 10 == 0:
            print(f"  embedded {done}/{len(texts)}")

    for start in range(0, len(chunks), BATCH_SIZE):
        add_chunks(
            collection,
            chunks[start : start + BATCH_SIZE],
            all_embeddings[start : start + BATCH_SIZE],
        )

    print(f"Index built — {collection.count()} documents in '{collection_name}'")
