from __future__ import annotations

import threading

import numpy as np

from rag_assistant.embeddings import Vector
from rag_assistant.models import Chunk, Filters
from rag_assistant.retrieval.bm25 import bm25_scores


class InMemoryVectorStore:
    """Exact (brute-force) cosine search with numpy. Suitable for tests and corpora up to ~100K chunks."""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, Vector] = {}
        self._lock = threading.Lock()

    def upsert(self, chunks: list[Chunk], vectors: list[Vector]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        with self._lock:
            for c, v in zip(chunks, vectors, strict=True):
                self._chunks[c.id] = c
                self._vectors[c.id] = v

    def delete_document(self, tenant: str, document_id: str) -> int:
        with self._lock:
            ids = [i for i, c in self._chunks.items() if c.tenant == tenant and c.document_id == document_id]
            for i in ids:
                del self._chunks[i], self._vectors[i]
            return len(ids)

    def _candidates(self, filters: Filters) -> list[Chunk]:
        with self._lock:
            return [c for c in self._chunks.values() if filters.allows(c)]

    def vector_search(self, vector: Vector, filters: Filters, k: int) -> list[tuple[Chunk, float]]:
        candidates = self._candidates(filters)
        if not candidates:
            return []
        matrix = np.stack([self._vectors[c.id] for c in candidates])
        scores = matrix @ vector
        order = np.argsort(-scores)[:k]
        return [(candidates[i], float(scores[i])) for i in order]

    def keyword_search(self, query: str, filters: Filters, k: int) -> list[tuple[Chunk, float]]:
        candidates = self._candidates(filters)
        scores = bm25_scores(query, [f"{c.title} {c.section} {c.text}" for c in candidates])
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda x: -x[1])
        return [(c, s) for c, s in ranked[:k] if s > 0]

    def documents(self, tenant: str) -> list[dict[str, object]]:
        docs: dict[str, dict[str, object]] = {}
        with self._lock:
            for c in self._chunks.values():
                if c.tenant == tenant:
                    d = docs.setdefault(
                        c.document_id, {"id": c.document_id, "title": c.title, "source": c.source, "chunks": 0}
                    )
                    d["chunks"] = int(str(d["chunks"])) + 1
        return sorted(docs.values(), key=lambda d: str(d["id"]))
