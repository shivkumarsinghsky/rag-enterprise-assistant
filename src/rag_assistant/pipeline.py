"""The RAG pipeline: ingestion and question answering.

question → guardrails → (condense with history) → hybrid retrieval with tenant/ACL filters
         → injection screening → context packing → prompt → LLM → citation validation → answer
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

from rag_assistant import metrics
from rag_assistant.citations import resolve_citations
from rag_assistant.config import Settings
from rag_assistant.conversation import ConversationStore
from rag_assistant.embeddings import Embedder, HashingEmbedder, OpenAICompatibleEmbedder
from rag_assistant.guardrails import check_question, looks_like_injection, redact_pii
from rag_assistant.ingestion.chunking import chunk_document, embedding_text
from rag_assistant.llm import LLM, ExtractiveAnswerer, OpenAICompatibleChat
from rag_assistant.models import Answer, Citation, Document, Filters, Principal
from rag_assistant.prompting import REFUSAL, build_prompt, pack_context
from rag_assistant.retrieval.retriever import HybridRetriever
from rag_assistant.vectorstore import VectorStore
from rag_assistant.vectorstore.memory import InMemoryVectorStore

log = logging.getLogger("rag.pipeline")


class RagPipeline:
    def __init__(
        self,
        settings: Settings,
        store: VectorStore,
        embedder: Embedder,
        llm: LLM,
        conversations: ConversationStore | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.retriever = HybridRetriever(store, embedder, settings.min_relevance)
        self.conversations = conversations or ConversationStore()

    # ---- ingestion -------------------------------------------------------------------------------------

    def ingest(self, doc: Document) -> int:
        """Idempotent: re-ingesting a document replaces its previous chunks."""
        chunks = chunk_document(doc, self.settings.chunk_size_words, self.settings.chunk_overlap_words)
        vectors = self.embedder.embed([embedding_text(c) for c in chunks])
        self.store.delete_document(doc.tenant, doc.id)
        self.store.upsert(chunks, vectors)
        metrics.INGESTED_CHUNKS.inc(len(chunks))
        log.info(
            "document ingested", extra={"fields": {"document": doc.id, "tenant": doc.tenant, "chunks": len(chunks)}}
        )
        return len(chunks)

    # ---- question answering ----------------------------------------------------------------------------

    def ask(
        self,
        principal: Principal,
        question: str,
        conversation_id: str | None = None,
        metadata_filter: dict[str, str] | None = None,
        document_ids: list[str] | None = None,
    ) -> Answer:
        timings: dict[str, float] = {}

        @contextmanager
        def stage(name: str) -> Iterator[None]:
            started = time.perf_counter()
            yield
            elapsed = time.perf_counter() - started
            timings[name] = round(elapsed * 1000, 2)
            metrics.STAGE_SECONDS.labels(stage=name).observe(elapsed)

        question = check_question(question)
        history = (
            self.conversations.history(principal.tenant, principal.user, conversation_id) if conversation_id else []
        )
        with stage("condense"):
            search_query = self.llm.condense(history, question) if history else question

        filters = Filters(
            tenant=principal.tenant,
            groups=principal.groups,
            document_ids=frozenset(document_ids) if document_ids else None,
            metadata=metadata_filter or {},
        )
        with stage("retrieve"):
            retrieved = self.retriever.retrieve(search_query, filters, self.settings.top_k, self.settings.candidate_k)

        flagged = [sc.chunk.id for sc in retrieved if looks_like_injection(sc.chunk.text)]
        metrics.FLAGGED.inc(len(flagged))
        safe = [sc for sc in retrieved if sc.chunk.id not in flagged]
        context = pack_context(safe, self.settings.context_budget_words)

        usage: dict[str, int] = {}
        citations: list[Citation] = []
        if not context:
            text, citations, grounded = REFUSAL, [], False
        else:
            with stage("generate"):
                generation = self.llm.generate(build_prompt(question, context, history))
            usage = generation.usage
            for kind, n in usage.items():
                metrics.TOKENS.labels(kind=kind).inc(n)
            text, citations, grounded = resolve_citations(generation.text, context)
            if not grounded and REFUSAL.lower() not in text.lower():
                # The model answered without citing anything: do not present unsupported claims as fact.
                text = REFUSAL

        if conversation_id:
            self.conversations.append(principal.tenant, principal.user, conversation_id, question, text)
        metrics.QUERIES.labels(grounded=str(grounded).lower()).inc()
        log.info(
            "question answered",
            extra={
                "fields": {
                    "tenant": principal.tenant,
                    "question": redact_pii(question)[:200],
                    "grounded": grounded,
                    "retrieved": [sc.chunk.id for sc in retrieved],
                    "flagged": flagged,
                    "timings_ms": timings,
                }
            },
        )
        return Answer(text, citations, grounded, retrieved, search_query, flagged, timings, usage)


def build_pipeline(settings: Settings) -> RagPipeline:
    embedder: Embedder
    if settings.embedding_provider == "openai":
        embedder = OpenAICompatibleEmbedder(
            settings.openai_base_url,
            settings.openai_api_key,
            settings.embedding_model,
            settings.embedding_dimensions,
            settings.request_timeout_seconds,
        )
    else:
        embedder = HashingEmbedder(settings.embedding_dimensions)

    llm: LLM
    if settings.llm_provider == "openai":
        llm = OpenAICompatibleChat(
            settings.openai_base_url,
            settings.openai_api_key,
            settings.llm_model,
            settings.llm_temperature,
            settings.llm_max_tokens,
            settings.request_timeout_seconds,
        )
    else:
        llm = ExtractiveAnswerer()

    store: VectorStore
    if settings.vector_store == "pgvector":
        from rag_assistant.vectorstore.pgvector import PgVectorStore

        store = PgVectorStore(settings.database_url, embedder.dimensions)
    else:
        store = InMemoryVectorStore()
    return RagPipeline(settings, store, embedder, llm)
