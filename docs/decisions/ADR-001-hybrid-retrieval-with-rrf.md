# ADR-001: Hybrid Retrieval (Vector + Keyword) With Reciprocal Rank Fusion

- **Status:** Accepted
- **Date:** 2026-10-01

## Context

Enterprise questions mix natural language ("how long does the seal replacement take") with exact identifiers
(`P-101`, `SEAL-KIT-40`, error codes). Dense embeddings handle paraphrase well but often rank exact identifiers
poorly; keyword search has the opposite behaviour.

## Decision

Run vector search and keyword search (BM25 in memory, PostgreSQL full-text with pgvector) over the same filtered
candidate set and fuse the ranked lists with Reciprocal Rank Fusion (k = 60).

## Alternatives Considered

- **Vector only** — simplest; misses identifiers.
- **Weighted score sum** — needs calibration between cosine similarity and BM25 scores, which drifts when
  the corpus or embedding model changes.
- **Cross-encoder re-ranking** — best precision; additional model and latency. Complementary: it can re-rank the
  fused candidates later.

## Trade-offs

Two searches per query; RRF ignores score magnitudes (a very strong single match is not boosted beyond rank 1).

## Consequences

Retrieval results expose per-retriever ranks (`debug.retrieved[].ranks`) so relevance problems can be diagnosed.
