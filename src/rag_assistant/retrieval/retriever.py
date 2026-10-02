"""Hybrid retrieval: vector search + keyword search fused with Reciprocal Rank Fusion (RRF).

- Vector search finds paraphrases ("pump shakes" ~ "excessive vibration").
- Keyword search finds exact identifiers ("P-101", "ERR-4412") that embeddings blur.
- RRF combines ranked lists without having to calibrate their incomparable scores: score = Σ 1 / (k + rank).
Filtering (tenant, ACL, metadata) is applied inside each search, before ranking.
"""

from __future__ import annotations

from rag_assistant.embeddings import Embedder
from rag_assistant.models import Chunk, Filters, ScoredChunk
from rag_assistant.vectorstore import VectorStore

RRF_K = 60


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder, min_vector_score: float = 0.05) -> None:
        self.store = store
        self.embedder = embedder
        self.min_vector_score = min_vector_score

    def retrieve(self, query: str, filters: Filters, k: int = 5, candidate_k: int = 20) -> list[ScoredChunk]:
        vector = self.embedder.embed([query])[0]
        vec_hits = [
            (c, s) for c, s in self.store.vector_search(vector, filters, candidate_k) if s >= self.min_vector_score
        ]
        kw_hits = self.store.keyword_search(query, filters, candidate_k)
        return fuse({"vector": vec_hits, "keyword": kw_hits}, k)


def fuse(result_lists: dict[str, list[tuple[Chunk, float]]], k: int) -> list[ScoredChunk]:
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    chunks: dict[str, Chunk] = {}
    for name, hits in result_lists.items():
        for rank, (chunk, _score) in enumerate(hits, start=1):
            chunks[chunk.id] = chunk
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (RRF_K + rank)
            ranks.setdefault(chunk.id, {})[name] = rank
    ordered = sorted(scores, key=lambda cid: -scores[cid])
    # Drop near-duplicate passages (same content hash, e.g. boilerplate repeated across documents).
    seen_hashes: set[str] = set()
    result: list[ScoredChunk] = []
    for cid in ordered:
        h = str(chunks[cid].metadata.get("content_hash", cid))
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        result.append(ScoredChunk(chunks[cid], scores[cid], ranks[cid]))
        if len(result) == k:
            break
    return result
