"""HTTP API. The tenant and groups come from the API key (a stand-in for a verified OIDC token), never from the
request body, so a caller cannot widen their own retrieval scope."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from rag_assistant.guardrails import GuardrailViolation
from rag_assistant.ingestion.loaders import load_text, title_from
from rag_assistant.models import Document, Principal
from rag_assistant.pipeline import RagPipeline


class DocumentIn(BaseModel):
    id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]{1,100}$")
    title: str | None = Field(default=None, max_length=300)
    content: str = Field(min_length=1, max_length=2_000_000)
    content_type: str = Field(default="text/markdown", pattern=r"^text/(markdown|plain|html)$")
    source: str = Field(default="api", max_length=500)
    acl: list[str] = Field(default_factory=list, max_length=50)
    metadata: dict[str, str] = Field(default_factory=dict)


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    filters: dict[str, str] = Field(default_factory=dict)
    document_ids: list[str] | None = None
    debug: bool = False


def get_principal(request: Request, x_api_key: Annotated[str | None, Header()] = None) -> Principal:
    principals: dict[str, Principal] = request.app.state.principals
    p = principals.get(hashlib.sha256((x_api_key or "").encode()).hexdigest())
    if p is None:
        raise HTTPException(401, "missing or invalid API key")
    return p


Caller = Annotated[Principal, Depends(get_principal)]


def create_app(pipeline: RagPipeline, api_keys: dict[str, dict[str, Any]]) -> FastAPI:
    app = FastAPI(title="RAG Enterprise Assistant", version="1.0.0")
    # Store only key hashes in memory: a heap dump or debug endpoint never reveals usable keys.
    app.state.principals = {
        hashlib.sha256(k.encode()).hexdigest(): Principal(v["tenant"], v["user"], frozenset(v.get("groups", [])))
        for k, v in api_keys.items()
    }

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready() -> dict[str, str]:
        return {"status": "ready", "embedder": pipeline.embedder.model_name, "llm": pipeline.llm.name}

    @app.get("/metrics", include_in_schema=False)
    def prometheus() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/documents", status_code=201)
    def ingest(body: DocumentIn, caller: Caller) -> dict[str, Any]:
        text = load_text(body.content, body.content_type)
        doc_id = body.id or uuid.uuid4().hex
        doc = Document(
            id=doc_id,
            tenant=caller.tenant,
            title=body.title or title_from(text, doc_id),
            text=text,
            source=body.source,
            acl=frozenset(body.acl),
            metadata=body.metadata,
        )
        return {"id": doc_id, "chunks": pipeline.ingest(doc)}

    @app.get("/v1/documents")
    def documents(caller: Caller) -> list[dict[str, object]]:
        return pipeline.store.documents(caller.tenant)

    @app.delete("/v1/documents/{document_id}", status_code=204)
    def delete(document_id: str, caller: Caller) -> Response:
        if not pipeline.store.delete_document(caller.tenant, document_id):
            raise HTTPException(404, "document not found")
        return Response(status_code=204)

    @app.post("/v1/conversations", status_code=201)
    def new_conversation(caller: Caller) -> dict[str, str]:
        return {"conversationId": pipeline.conversations.new_id()}

    @app.post("/v1/query")
    def query(body: QueryIn, caller: Caller) -> dict[str, Any]:
        try:
            answer = pipeline.ask(caller, body.question, body.conversation_id, body.filters, body.document_ids)
        except GuardrailViolation as e:
            raise HTTPException(400, str(e)) from e
        result: dict[str, Any] = {
            "answer": answer.text,
            "grounded": answer.grounded,
            "citations": [c.__dict__ for c in answer.citations],
        }
        if body.debug:
            result["debug"] = {
                "rewrittenQuestion": answer.rewritten_question,
                "retrieved": [
                    {
                        "chunkId": sc.chunk.id,
                        "score": round(sc.score, 5),
                        "ranks": sc.ranks,
                        "section": sc.chunk.section,
                    }
                    for sc in answer.retrieved
                ],
                "flaggedChunks": answer.flagged_chunks,
                "timingsMs": answer.timings_ms,
                "usage": answer.usage,
            }
        return result

    return app


def parse_api_keys(raw: str) -> dict[str, dict[str, Any]]:
    keys: dict[str, dict[str, Any]] = json.loads(raw)
    for key, value in keys.items():
        if len(key) < 12 or "tenant" not in value or "user" not in value:
            raise ValueError("each API key needs >= 12 chars and 'tenant' and 'user' fields")
    return keys
