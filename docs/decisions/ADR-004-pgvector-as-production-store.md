# ADR-004: PostgreSQL + pgvector as the Production Vector Store

- **Status:** Accepted
- **Date:** 2026-10-01

## Context

The store must support ANN vector search, keyword search, rich metadata filtering, ACL predicates, transactional
updates and backups — and fit an enterprise that already operates PostgreSQL.

## Decision

Use PostgreSQL with pgvector (HNSW index, cosine distance) and built-in full-text search (GIN index), with tenant,
ACL and metadata predicates in the same SQL statement.

## Alternatives Considered

| Option | Strengths | Weaknesses |
|---|---|---|
| Dedicated vector DB (Qdrant, Weaviate, Pinecone, Milvus) | Very large scale, rich ANN features | Another system to secure, back up and keep in sync with metadata |
| Elasticsearch/OpenSearch | Mature BM25 + kNN + filtering | Heavier operations; JVM cluster |
| In-memory | Zero setup | Not persistent; single process |

## Trade-offs

Filtered HNSW search can degrade when filters are very selective (many candidates discarded); mitigations are
partial indexes per tenant, iterative scans, or a dedicated store at very large scale.

## Consequences

One database for chunks, metadata and (optionally) conversation history; standard backup and access control.
