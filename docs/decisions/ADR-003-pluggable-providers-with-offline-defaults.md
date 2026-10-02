# ADR-003: Pluggable Providers With Offline Defaults

- **Status:** Accepted
- **Date:** 2026-10-01

## Context

The reference must run in CI and on a laptop without API keys, while supporting hosted and self-hosted models in
real use. Hard-coding one vendor SDK ties the architecture to that vendor.

## Decision

- Small `Embedder`, `LLM` and `VectorStore` protocols.
- Providers: OpenAI-compatible HTTP APIs for embeddings and chat (works with OpenAI, Ollama, vLLM, LiteLLM, and Azure
  OpenAI behind a compatible gateway); in-memory and pgvector stores.
- Offline defaults: a deterministic **hashing embedder** and an **extractive answerer** that composes cited answers
  from context sentences. They are clearly labelled as baselines, not language models.

## Alternatives Considered

- **LangChain/LlamaIndex abstractions** — fast to start; heavier dependency surface and less visible behaviour for a
  reference whose purpose is to show the mechanics. The agent platform repository uses LangGraph where orchestration
  is the point.
- **Vendor SDKs** — richer features (streaming helpers, structured outputs) at the cost of portability.

## Trade-offs

The offline baseline is lexical, not semantic; evaluation numbers in offline mode do not represent quality with real
models.

## Consequences

The full pipeline, citations, guardrails and evaluation run in CI with no secrets; switching to real models is
configuration only.
