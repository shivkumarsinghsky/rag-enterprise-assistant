from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Principal:
    """Who is asking. Derived from the API credential, never from the request body."""

    tenant: str
    user: str
    groups: frozenset[str]


@dataclass(frozen=True)
class Document:
    id: str
    tenant: str
    title: str
    text: str
    source: str
    #: Groups allowed to read the document; empty means every user of the tenant.
    acl: frozenset[str] = frozenset()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    tenant: str
    title: str
    section: str
    text: str
    ordinal: int
    source: str
    acl: frozenset[str] = frozenset()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Filters:
    """Retrieval filters. `tenant` and `groups` come from the principal; the rest from the request."""

    tenant: str
    groups: frozenset[str]
    document_ids: frozenset[str] | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def allows(self, chunk: Chunk) -> bool:
        if chunk.tenant != self.tenant:
            return False
        if chunk.acl and not (chunk.acl & self.groups):
            return False
        if self.document_ids is not None and chunk.document_id not in self.document_ids:
            return False
        return all(str(chunk.metadata.get(k)) == v for k, v in self.metadata.items())


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    #: Per-retriever ranks for explainability, e.g. {"vector": 1, "keyword": 3}
    ranks: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Citation:
    index: int
    chunk_id: str
    document_id: str
    title: str
    section: str
    source: str


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Citation]
    grounded: bool
    retrieved: list[ScoredChunk]
    rewritten_question: str
    flagged_chunks: list[str]
    timings_ms: dict[str, float]
    usage: dict[str, int]
