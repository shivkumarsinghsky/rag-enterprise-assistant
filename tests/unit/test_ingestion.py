import pytest

from rag_assistant.ingestion.chunking import chunk_document, embedding_text, split_sections
from rag_assistant.ingestion.loaders import html_to_text, load_text, normalise, title_from
from rag_assistant.models import Document

MD = """# Pump Manual

Intro paragraph.

## Maintenance

### Lubrication

Grease every 2000 hours.

## Safety

Use LOTO-07.
"""


def test_sections_keep_heading_paths():
    sections = split_sections(MD)
    assert [s.path for s in sections] == [
        "Pump Manual",
        "Pump Manual > Maintenance > Lubrication",
        "Pump Manual > Safety",
    ]


def test_chunks_respect_size_and_overlap():
    words = " ".join(f"w{i}." for i in range(500))
    doc = Document("d1", "acme", "Doc", f"# Doc\n\n{words}", "src")
    chunks = chunk_document(doc, size=100, overlap=20)
    assert all(len(c.text.split()) <= 100 for c in chunks)
    assert len(chunks) >= 5
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-20:] == second[:20]  # overlap carries context across the boundary
    assert [c.id for c in chunks[:2]] == ["d1:0", "d1:1"]
    assert chunks[0].metadata["content_hash"] != chunks[1].metadata["content_hash"]


def test_chunk_carries_tenant_acl_and_metadata():
    doc = Document("d", "acme", "T", MD, "s", frozenset({"hr"}), {"department": "hr"})
    c = chunk_document(doc, size=50, overlap=5)[0]
    assert (c.tenant, c.acl, c.metadata["department"]) == ("acme", frozenset({"hr"}), "hr")
    assert embedding_text(c).startswith("T\nPump Manual")


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_document(Document("d", "t", "T", MD, "s"), size=10, overlap=10)


def test_html_and_normalisation():
    html = "<html><script>x()</script><h1>Title</h1><p>Hello&nbsp;  world</p><ul><li>one</li></ul></html>"
    text = load_text(html, "text/html")
    assert text.startswith("# Title")
    assert "Hello world" in text and "x()" not in text
    assert normalise("a\u0000b\r\n\n\n\nc") == "ab\n\nc"
    assert title_from("# Real Title\nbody", "fallback") == "Real Title"
    assert "Title" in html_to_text("<h2>Title</h2>")
