"""PgVectorStore against a real PostgreSQL with the pgvector extension.

Skipped unless TEST_PGVECTOR_URL is set (a disposable database; the rag_chunks table is dropped)."""

import os

import pytest

from rag_assistant.embeddings import HashingEmbedder
from rag_assistant.models import Chunk, Filters
from rag_assistant.retrieval.retriever import HybridRetriever

URL = os.environ.get("TEST_PGVECTOR_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_PGVECTOR_URL not set")


@pytest.fixture
def store():
    import psycopg

    from rag_assistant.vectorstore.pgvector import PgVectorStore

    with psycopg.connect(URL, autocommit=True) as c:
        c.execute("DROP TABLE IF EXISTS rag_chunks")
    return PgVectorStore(URL, 128)


def chunk(cid: str, text: str, tenant: str = "acme", acl: frozenset[str] = frozenset(), **meta: str) -> Chunk:
    return Chunk(
        cid, cid.split(":")[0], tenant, "Manual", "Section", text, 0, "src.md", acl, {"content_hash": cid, **meta}
    )


def test_filtered_vector_and_keyword_search(store):
    e = HashingEmbedder(128)
    chunks = [
        chunk("pump:0", "Pump P-101 inspection every 90 days", department="maintenance"),
        chunk("pump:1", "Seal kit SEAL-KIT-40 replacement takes 4 hours", department="maintenance"),
        chunk("bonus:0", "Pump bonus pool is 12 percent", acl=frozenset({"executives"}), department="hr"),
        chunk("other:0", "Pump inspection every 30 days", tenant="globex"),
    ]
    store.upsert(chunks, e.embed([c.text for c in chunks]))
    staff = Filters("acme", frozenset({"maintenance"}))
    vec = store.vector_search(e.embed(["pump inspection"])[0], staff, 10)
    assert {c.id for c, _ in vec} == {"pump:0", "pump:1"}
    kw = store.keyword_search("SEAL-KIT-40 replacement", staff, 10)
    assert kw[0][0].id == "pump:1"
    assert store.keyword_search("'); DROP TABLE rag_chunks; --", staff, 5) == []
    hr = Filters("acme", frozenset({"executives"}), metadata={"department": "hr"})
    assert [c.id for c, _ in store.vector_search(e.embed(["bonus"])[0], hr, 10)] == ["bonus:0"]
    hits = HybridRetriever(store, e, 0.0).retrieve("pump inspection interval", staff, k=3)
    assert hits[0].chunk.id == "pump:0" and {h.chunk.tenant for h in hits} == {"acme"}


def test_upsert_is_idempotent_and_delete_is_tenant_scoped(store):
    e = HashingEmbedder(128)
    c = chunk("doc:0", "first version")
    store.upsert([c], e.embed([c.text]))
    c2 = chunk("doc:0", "second version")
    store.upsert([c2], e.embed([c2.text]))
    assert store.documents("acme") == [{"id": "doc", "title": "Manual", "source": "src.md", "chunks": 1}]
    assert store.delete_document("globex", "doc") == 0
    assert store.delete_document("acme", "doc") == 1
