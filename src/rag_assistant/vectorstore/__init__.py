from __future__ import annotations

from typing import Protocol

from rag_assistant.embeddings import Vector
from rag_assistant.models import Chunk, Filters


class VectorStore(Protocol):
    """Storage for chunks + embeddings. Every search takes Filters so tenant/ACL filtering happens inside the
    store, before ranking — never as a post-filter that could leak counts or starve results."""

    def upsert(self, chunks: list[Chunk], vectors: list[Vector]) -> None: ...

    def delete_document(self, tenant: str, document_id: str) -> int: ...

    def vector_search(self, vector: Vector, filters: Filters, k: int) -> list[tuple[Chunk, float]]: ...

    def keyword_search(self, query: str, filters: Filters, k: int) -> list[tuple[Chunk, float]]: ...

    def documents(self, tenant: str) -> list[dict[str, object]]: ...
