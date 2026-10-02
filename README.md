# RAG Enterprise Assistant — Retrieval-Augmented Generation Reference Implementation

[![CI](https://github.com/shivkumarsinghsky/rag-enterprise-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/shivkumarsinghsky/rag-enterprise-assistant/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![pgvector](https://img.shields.io/badge/PostgreSQL-pgvector-336791)
![License](https://img.shields.io/badge/license-MIT-green)

An enterprise **Retrieval-Augmented Generation (RAG)** reference implementation by **Shiv Kumar**, built with
Python and FastAPI. It answers questions over company documents with **citations**, and only uses content the
asking user is allowed to read.

It covers the full pipeline:

- document ingestion and structure-aware chunking;
- embeddings and hybrid retrieval (vector + keyword with Reciprocal Rank Fusion), with tenant, ACL and metadata
  filtering;
- prompt construction with grounding rules, plus prompt-injection screening;
- the LLM call, with citation validation afterwards;
- conversation context;
- an evaluation harness that runs in CI.

It runs **fully offline by default** (deterministic hashing embeddings and an extractive answerer), so the
pipeline, tests and evaluation need no API keys. Any **OpenAI-compatible** endpoint can be plugged in instead —
OpenAI, Azure OpenAI behind a compatible gateway, **Ollama** or vLLM for local models.

> Reference / portfolio implementation. The offline providers are baselines for testing, not substitutes for real
> embedding and language models.

## Architecture

```mermaid
flowchart TB
    Docs["Documents<br/>+ ACL + metadata"] --> Ingest["Ingestion<br/>load, normalise"]
    Ingest --> Chunk["Chunking<br/>sections, paragraphs, overlap"]
    Chunk --> Embed["Embeddings"]
    Embed --> Store[("Vector store<br/>memory or pgvector")]
    User["User + credential"] --> API["FastAPI"]
    API --> Ret["Retriever<br/>vector + keyword, RRF<br/>tenant and ACL filters"]
    Store --> Ret
    Ret --> Guard["Injection screening<br/>context packing"]
    Guard --> LLM["LLM<br/>grounded prompt"]
    LLM --> Cite["Citation validation"]
    Cite --> Resp["Response + citations"]
```

```text
Documents → Ingestion → Chunking → Embeddings → Vector Store → Retriever → LLM → Response + Citations
```

The full design — chunking and retrieval strategies, data isolation, hallucination mitigation, evaluation, latency
and cost — is in [docs/rag-architecture.md](docs/rag-architecture.md).

## Key Capabilities

| Capability | Implementation |
|---|---|
| Document ingestion | Markdown, text, HTML; normalisation; idempotent re-ingestion (`ingestion/loaders.py`) |
| Chunking | Heading-path aware sections → paragraphs → sentences, configurable size/overlap (`ingestion/chunking.py`) |
| Embeddings | Offline hashing embedder; OpenAI-compatible `/v1/embeddings` (`embeddings.py`) |
| Vector search | In-memory cosine (numpy); PostgreSQL **pgvector** with HNSW (`vectorstore/`) |
| Keyword search | BM25 (memory); PostgreSQL full-text with GIN (pgvector mode) |
| Hybrid retrieval | Reciprocal Rank Fusion, near-duplicate removal (`retrieval/retriever.py`) |
| Metadata filtering | Tenant + ACL groups from the credential; metadata and document filters from the request |
| Prompt construction | Grounding rules, delimited sources, context budget, bounded history (`prompting.py`) |
| LLM response | OpenAI-compatible chat with timeouts and retries; offline extractive answerer (`llm.py`) |
| Citations | `[n]` markers validated against provided sources; uncited answers replaced by a refusal (`citations.py`) |
| Conversation context | Per-(tenant, user) memory; follow-ups condensed into standalone queries (`conversation.py`) |
| Guardrails | Prompt-injection screening of retrieved chunks, PII redaction in logs, input limits (`guardrails.py`) |
| Evaluation | Hit rate, MRR, citation precision, answer recall, refusal accuracy (`evaluation.py`) |

## Technology Stack

| Area | Choice |
|---|---|
| Language / API | Python 3.10+, FastAPI, Uvicorn, pydantic v2 |
| Vector store | PostgreSQL 16 + pgvector (HNSW, cosine) + full-text search; in-memory for tests |
| Models | Any OpenAI-compatible API (OpenAI, Ollama, vLLM, LiteLLM); offline baselines |
| HTTP client | httpx |
| Observability | JSON logs, Prometheus metrics, per-stage timings |
| Tests | pytest (unit, API, pgvector integration), evaluation gate in CI |

## Repository Structure

```text
rag-enterprise-assistant/
├── src/rag_assistant/
│   ├── ingestion/          # loaders, chunking
│   ├── retrieval/          # BM25, hybrid retriever (RRF)
│   ├── vectorstore/        # protocol, in-memory, pgvector
│   ├── embeddings.py  llm.py  prompting.py  citations.py  guardrails.py  conversation.py
│   ├── pipeline.py         # ingestion + question answering
│   ├── api.py              # FastAPI
│   ├── evaluation.py       # metrics over a labelled question set
│   └── __main__.py         # CLI: serve | ingest | ask | eval
├── sample_docs/<tenant>/   # illustrative corpus (manuals, procedures, policies, a confidential doc, an injection attempt)
├── eval/questions.jsonl    # labelled evaluation questions
├── tests/                  # unit, API and pgvector tests
├── docker/  docker-compose.yml
└── docs/                   # RAG architecture, ADRs
```

## Getting Started

```bash
git clone https://github.com/shivkumarsinghsky/rag-enterprise-assistant.git
cd rag-enterprise-assistant
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Ask a question over the sample corpus (offline providers)
python -m rag_assistant ask acme "What grease is used for P-101 bearings and how much?" --groups maintenance
```

```text
Bearings are lubricated every 2000 running hours with lithium-complex grease, 40 grams per bearing. [1] ...
  [1] Boiler Feed Pump P-101 Maintenance Manual — ... > Preventive Maintenance (acme/pump-p101-manual.md)
```

### Run the API

```bash
cp .env.example .env
python -m rag_assistant serve          # ingests sample_docs at startup; http://localhost:8000/docs
```

### Docker (pgvector)

```bash
docker compose up -d --build           # PostgreSQL + pgvector and the API on :8000
```

### Use real models

```bash
# OpenAI
EMBEDDING_PROVIDER=openai EMBEDDING_DIMENSIONS=1536 LLM_PROVIDER=openai OPENAI_API_KEY=sk-... python -m rag_assistant serve

# Local models via Ollama (after: ollama pull nomic-embed-text && ollama pull llama3.1)
EMBEDDING_PROVIDER=openai OPENAI_BASE_URL=http://localhost:11434/v1 EMBEDDING_MODEL=nomic-embed-text \
EMBEDDING_DIMENSIONS=768 LLM_PROVIDER=openai LLM_MODEL=llama3.1 python -m rag_assistant serve
```

## Configuration

See [`.env.example`](.env.example). Main settings:

| Variable | Purpose | Default |
|---|---|---|
| `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS` | Embeddings | `hashing`, 384 dims |
| `LLM_PROVIDER`, `LLM_MODEL`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS` | Generation | `extractive` |
| `OPENAI_BASE_URL`, `OPENAI_API_KEY` | Any OpenAI-compatible endpoint | OpenAI |
| `VECTOR_STORE`, `DATABASE_URL` | `memory` or `pgvector` | `memory` |
| `CHUNK_SIZE_WORDS`, `CHUNK_OVERLAP_WORDS` | Chunking | 180 / 30 |
| `TOP_K`, `CANDIDATE_K`, `CONTEXT_BUDGET_WORDS` | Retrieval and context | 5 / 20 / 1200 |
| `API_KEYS` | API key → tenant, user, groups (development stand-in for OIDC) | one dev key |

No real secrets are committed; `.env` is git-ignored.

## API Examples

```http
POST /v1/documents
X-API-Key: dev-acme-maintenance
{ "id": "chiller-c1", "content": "# Chiller C1\n\n## Limits\n\nCondenser pressure limit is 14 bar.",
  "content_type": "text/markdown", "acl": ["maintenance"], "metadata": {"department": "maintenance"} }
→ 201 { "id": "chiller-c1", "chunks": 1 }

POST /v1/query
X-API-Key: dev-acme-maintenance
{ "question": "What is the condenser pressure limit of Chiller C1?", "filters": {"department": "maintenance"}, "debug": true }
→ 200 { "answer": "Condenser pressure limit is 14 bar. [1]", "grounded": true,
        "citations": [{ "index": 1, "document_id": "chiller-c1", "section": "Chiller C1 > Limits", ... }],
        "debug": { "retrieved": [{ "chunkId": "chiller-c1:0", "ranks": {"vector": 1, "keyword": 1} }],
                   "flaggedChunks": [], "timingsMs": { "retrieve": 0.4, "generate": 0.1 } } }

POST /v1/conversations                      → { "conversationId": "..." }   (pass it to /v1/query for follow-ups)
GET  /v1/documents   |   DELETE /v1/documents/{id}
GET  /health/live    |   /health/ready   |   /metrics
```

## Testing and Evaluation

```bash
pytest                                       # unit + API tests (pgvector tests skip without a database)
TEST_PGVECTOR_URL=postgresql://rag:rag-local-dev@localhost:5432/rag pytest   # include pgvector tests
ruff check . && ruff format --check . && mypy
python -m rag_assistant eval eval/questions.jsonl --min-hit-rate 0.9
```

Tests cover chunking and overlap, HTML loading, stemming/tokenisation, BM25, RRF fusion, tenant/ACL/metadata
filters, prompt-injection detection, citation validation, LLM retries (mocked HTTP), conversation scoping and
follow-ups, tenant isolation and ACL denial end-to-end, and the HTTP API (auth, isolation, deletes). With
pgvector they also cover filtered ANN search, keyword search, tsquery input sanitisation and idempotent upserts.

The evaluation gate runs in CI. The bundled set has 10 questions, including refusals for a confidential document,
another tenant's data and an off-topic question. On the sample corpus with the offline providers it currently
scores 1.0 on every metric. That reflects a tiny, hand-made set; it is not a quality benchmark.

## Docker

`docker/Dockerfile` (multi-stage, non-root) runs `python -m rag_assistant serve`. `docker-compose.yml` starts
PostgreSQL with pgvector and the API; the `ollama` profile adds a local model server.

## Architecture Decisions

| ADR | Decision |
|---|---|
| [ADR-001](docs/decisions/ADR-001-hybrid-retrieval-with-rrf.md) | Hybrid retrieval with Reciprocal Rank Fusion |
| [ADR-002](docs/decisions/ADR-002-authorization-in-retrieval.md) | Authorization enforced in retrieval, scope from the credential |
| [ADR-003](docs/decisions/ADR-003-pluggable-providers-with-offline-defaults.md) | Pluggable providers with offline defaults |
| [ADR-004](docs/decisions/ADR-004-pgvector-as-production-store.md) | PostgreSQL + pgvector as the production store |

## Scalability Considerations

- Stateless API replicas; the vector store holds all state (conversation memory would move to Redis/PostgreSQL
  for multi-replica deployments).
- HNSW and GIN indexes; tenant-leading indexes; per-tenant partitions or indexes for very large tenants.
- Ingestion is batch/asynchronous in production (queue + workers); embeddings are cached by content hash.
- Model calls dominate latency and cost: context budgets, smaller models for condensation, and caching.

## Reliability

Timeouts and bounded retries on model calls (429/5xx); refusal instead of unsupported answers; idempotent
ingestion; validation of all inputs; health and readiness endpoints.

## Security

- **Authorization in retrieval**: tenant and ACL predicates in every search; the request can only narrow scope.
- **Prompt-injection screening** of retrieved content plus delimited, untrusted-marked context; a test corpus
  document contains an injection attempt that never reaches the model.
- **Data protection**: PII redacted from logs; API keys held only as hashes; provider keys only via environment.
- **Not implemented** (reference scope): OIDC authentication, ACL synchronisation from source systems, rate limiting
  — see [enterprise-saas-platform](https://github.com/shivkumarsinghsky/enterprise-saas-platform) for those
  patterns.

## Observability

JSON logs per question (tenant, redacted question, retrieved and flagged chunk ids, grounded flag, per-stage
timings). Prometheus metrics: `rag_queries_total{grounded}`, `rag_stage_seconds{stage}`,
`rag_llm_tokens_total{kind}`, `rag_flagged_chunks_total`, `rag_ingested_chunks_total`. `debug: true` returns
retrieval ranks and timings for troubleshooting relevance.

## Future Improvements

Not implemented yet:

- Cross-encoder re-ranking; query expansion; parent–child retrieval.
- Streaming responses (SSE); structured outputs for answer + citations.
- PDF/DOCX loaders with table extraction; asynchronous ingestion workers.
- OIDC authentication, ACL sync connectors, per-tenant rate and token budgets.
- LLM-as-judge faithfulness evaluation; a larger evaluation set from real traffic.

## Related Projects

- [Enterprise AI Agent Platform](https://github.com/shivkumarsinghsky/enterprise-ai-agent-platform) — agents that use this kind of retrieval as a tool
- [System Design Architecture](https://github.com/shivkumarsinghsky/system-design-architecture) — [Enterprise AI Platform design](https://github.com/shivkumarsinghsky/system-design-architecture/blob/main/docs/designs/12-enterprise-ai-platform.md)
- [Enterprise SaaS Platform](https://github.com/shivkumarsinghsky/enterprise-saas-platform) — tenant isolation and RBAC
- [EAM Platform Architecture](https://github.com/shivkumarsinghsky/eam-platform-architecture) — the maintenance domain used in the sample corpus

## Author

**Shiv Kumar** — Senior Software Engineer / Software Architect
GitHub: [github.com/shivkumarsinghsky](https://github.com/shivkumarsinghsky)

## License

[MIT](LICENSE)
