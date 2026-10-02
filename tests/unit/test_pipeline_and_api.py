from fastapi.testclient import TestClient

from rag_assistant.api import create_app, parse_api_keys
from rag_assistant.evaluation import evaluate
from rag_assistant.prompting import REFUSAL

from ..conftest import EXEC, GLOBEX, MAINTENANCE, ROOT, STAFF


def test_grounded_answer_with_citations(pipeline):
    a = pipeline.ask(MAINTENANCE, "What vibration level requires switching P-101 to the standby pump?")
    assert a.grounded and "7.1 mm/s" in a.text
    assert {c.document_id for c in a.citations} == {"pump-p101-manual"}
    assert set(a.timings_ms) >= {"retrieve", "generate"}


def test_acl_hides_documents_from_users_without_access(pipeline):
    staff = pipeline.ask(STAFF, "What is the executive bonus pool for 2026?")
    assert not staff.grounded and staff.text == REFUSAL
    assert all(sc.chunk.document_id != "executive-compensation-2026" for sc in staff.retrieved)
    executive = pipeline.ask(EXEC, "What is the executive bonus pool for 2026?")
    assert executive.grounded and "12 percent" in executive.text


def test_tenants_are_isolated(pipeline):
    a = pipeline.ask(GLOBEX, "How often are feed pumps inspected?")
    assert "30 days" in a.text
    assert {sc.chunk.tenant for sc in a.retrieved} == {"globex"}


def test_injected_content_is_excluded_and_reported(pipeline):
    a = pipeline.ask(MAINTENANCE, "How long does the P-101 seal replacement take?")
    assert "vendor-newsletter:1" in a.flagged_chunks
    assert "10 minutes" not in a.text and "4 hours" in a.text


def test_follow_up_questions_use_conversation_history(pipeline):
    cid = pipeline.conversations.new_id()
    pipeline.ask(MAINTENANCE, "Which seal kit is needed for P-101?", conversation_id=cid)
    follow_up = pipeline.ask(MAINTENANCE, "How long does replacing it take?", conversation_id=cid)
    assert "P-101" in follow_up.rewritten_question
    assert "4 hours" in follow_up.text
    # another user cannot read this conversation
    assert pipeline.conversations.history("acme", "bob", cid) == []


def test_metadata_filters(pipeline):
    a = pipeline.ask(MAINTENANCE, "What is the daily meal allowance?", metadata_filter={"department": "maintenance"})
    assert not a.grounded


def test_bundled_evaluation_set_passes(pipeline):
    report = evaluate(pipeline, ROOT / "eval" / "questions.jsonl")
    assert report.hit_rate == 1.0 and report.refusal_accuracy == 1.0, report.failures


KEYS = {
    "acme-maint-key-1": {"tenant": "acme", "user": "alice", "groups": ["maintenance"]},
    "globex-key-0001": {"tenant": "globex", "user": "gina", "groups": []},
}


def test_api_end_to_end(pipeline):
    client = TestClient(create_app(pipeline, KEYS))
    acme, globex = {"x-api-key": "acme-maint-key-1"}, {"x-api-key": "globex-key-0001"}
    assert client.post("/v1/query", json={"question": "x"}).status_code == 401
    assert client.get("/v1/documents", headers={"x-api-key": "wrong-key-123"}).status_code == 401

    doc = {"id": "chiller-c1", "content": "# Chiller C1\n\n## Limits\n\nCondenser pressure limit is 14 bar.", "acl": []}
    assert client.post("/v1/documents", json=doc, headers=globex).json() == {"id": "chiller-c1", "chunks": 1}
    assert "chiller-c1" in [d["id"] for d in client.get("/v1/documents", headers=globex).json()]
    assert "chiller-c1" not in [d["id"] for d in client.get("/v1/documents", headers=acme).json()]

    r = client.post(
        "/v1/query",
        json={"question": "What is the condenser pressure limit of Chiller C1?", "debug": True},
        headers=globex,
    ).json()
    assert "14 bar" in r["answer"] and r["grounded"] and r["citations"][0]["document_id"] == "chiller-c1"
    assert r["debug"]["retrieved"][0]["chunkId"] == "chiller-c1:0"
    leak = client.post(
        "/v1/query", json={"question": "What is the condenser pressure limit of Chiller C1?"}, headers=acme
    ).json()
    assert not leak["grounded"]

    assert client.delete("/v1/documents/chiller-c1", headers=acme).status_code == 404  # not yours
    assert client.delete("/v1/documents/chiller-c1", headers=globex).status_code == 204
    cid = client.post("/v1/conversations", headers=acme).json()["conversationId"]
    assert (
        client.post(
            "/v1/query", json={"question": "Seal kit for P-101?", "conversation_id": cid}, headers=acme
        ).status_code
        == 200
    )
    assert client.post("/v1/query", json={"question": " " * 5}, headers=acme).status_code == 400
    assert "rag_queries_total" in client.get("/metrics").text


def test_api_key_validation():
    assert parse_api_keys('{"long-enough-key": {"tenant": "t", "user": "u"}}')
    import pytest

    with pytest.raises(ValueError):
        parse_api_keys('{"short": {"tenant": "t", "user": "u"}}')
