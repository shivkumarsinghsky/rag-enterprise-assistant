# Changelog

All notable changes to this repository are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-10-02

### Added

- Ingestion: Markdown/text/HTML loading, normalisation, structure-aware chunking with overlap and heading paths.
- Embeddings: offline hashing embedder, OpenAI-compatible embeddings.
- Stores: in-memory (BM25 + cosine) and PostgreSQL/pgvector (HNSW + full-text).
- Hybrid retrieval with RRF, tenant/ACL/metadata filtering, near-duplicate removal.
- Guardrails (prompt-injection screening, PII redaction, input limits), grounded prompting, citation validation.
- LLMs: OpenAI-compatible chat with retries; offline extractive answerer.
- Conversation memory with question condensation; FastAPI API; evaluation CLI and CI gate.
- Tests (unit, API, pgvector integration), Docker, Compose (pgvector, optional Ollama), docs and ADR-001 to ADR-004.
