import numpy as np

from rag_assistant.embeddings import HashingEmbedder, stem, tokenize
from rag_assistant.models import Chunk, Filters
from rag_assistant.retrieval.bm25 import bm25_scores
from rag_assistant.retrieval.retriever import HybridRetriever, fuse
from rag_assistant.vectorstore.memory import InMemoryVectorStore


def chunk(cid: str, text: str, tenant: str = "acme", acl: frozenset[str] = frozenset(), **meta: str) -> Chunk:
    return Chunk(cid, cid.split(":")[0], tenant, "T", "S", text, 0, "src", acl, {"content_hash": cid, **meta})


def test_tokenize_stems_and_drops_stopwords():
    assert tokenize("How often is the Inspection performed?") == ["often", "inspect", "perform"]
    assert stem("replacement") == stem("replacing") == "replac"
    assert stem("removed") == stem("remove") == "remov"
    assert stem("p-101") == "p-101"


def test_hashing_embedder_is_deterministic_normalised_and_lexically_meaningful():
    e = HashingEmbedder(256)
    a, b, c = e.embed(["pump seal replacement", "replace the pump seal", "hotel expense policy"])
    assert np.isclose(np.linalg.norm(a), 1.0)
    assert np.allclose(a, e.embed(["pump seal replacement"])[0])
    assert float(a @ b) > float(a @ c)


def test_bm25_prefers_exact_identifiers():
    docs = ["generic error handling guide", "error codes overview", "ERR-4412 means low suction pressure error"]
    scores = bm25_scores("error ERR-4412", docs)
    assert scores.index(max(scores)) == 2
    assert bm25_scores("", ["x"]) == [0.0]


def test_filters_enforce_tenant_acl_and_metadata():
    f = Filters("acme", frozenset({"maintenance"}), metadata={"department": "maintenance"})
    assert f.allows(chunk("a:0", "x", department="maintenance"))
    assert not f.allows(chunk("a:0", "x", tenant="globex", department="maintenance"))
    assert not f.allows(chunk("a:0", "x", acl=frozenset({"executives"}), department="maintenance"))
    assert f.allows(chunk("a:0", "x", acl=frozenset({"maintenance"}), department="maintenance"))
    assert not f.allows(chunk("a:0", "x", department="hr"))
    assert not Filters("acme", frozenset(), document_ids=frozenset({"b"})).allows(chunk("a:0", "x"))


def test_rrf_fusion_rewards_agreement_and_dedupes():
    a, b, c = chunk("a:0", "a"), chunk("b:0", "b"), chunk("c:0", "c")
    dup = Chunk("d:0", "d", "acme", "T", "S", "a", 0, "src", frozenset(), {"content_hash": "a:0"})
    fused = fuse({"vector": [(a, 0.9), (b, 0.8), (dup, 0.7)], "keyword": [(b, 5.0), (c, 1.0)]}, k=3)
    assert [s.chunk.id for s in fused] == ["b:0", "a:0", "c:0"]
    assert fused[0].ranks == {"vector": 2, "keyword": 1}


def test_store_search_never_returns_filtered_chunks():
    store = InMemoryVectorStore()
    e = HashingEmbedder(128)
    chunks = [
        chunk("pub:0", "pump inspection every 90 days"),
        chunk("sec:0", "pump inspection secret bonus", acl=frozenset({"executives"})),
        chunk("other:0", "pump inspection every 30 days", tenant="globex"),
    ]
    store.upsert(chunks, e.embed([c.text for c in chunks]))
    retriever = HybridRetriever(store, e, 0.0)
    hits = retriever.retrieve("pump inspection", Filters("acme", frozenset({"maintenance"})), k=5)
    assert [h.chunk.id for h in hits] == ["pub:0"]
    assert store.delete_document("acme", "pub") == 1
    assert store.documents("acme") == [{"id": "sec", "title": "T", "source": "src", "chunks": 1}]
