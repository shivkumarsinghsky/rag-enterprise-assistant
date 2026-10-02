"""Structure-aware chunking.

Strategy (see docs/rag-architecture.md#chunking-strategies):
1. Split the document into sections by Markdown headings and keep the heading path ("Pump P-101 > Maintenance")
   as chunk metadata — it is prepended to the embedded text so a chunk is understandable on its own.
2. Within a section, pack whole paragraphs into chunks up to `size` words.
3. Paragraphs longer than `size` are split on sentence boundaries, and only then on words.
4. Consecutive chunks in a section overlap by `overlap` words so facts spanning a boundary are not lost.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from rag_assistant.models import Chunk, Document

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass(frozen=True)
class Section:
    path: str
    paragraphs: list[str]


def split_sections(text: str) -> list[Section]:
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    current: list[str] = []
    buffer: list[str] = []

    def flush_paragraph() -> None:
        if buffer:
            current.append(" ".join(buffer).strip())
            buffer.clear()

    def flush_section() -> None:
        flush_paragraph()
        if current:
            sections.append(Section(" > ".join(t for _, t in stack), list(current)))
            current.clear()

    for line in text.splitlines():
        m = HEADING.match(line.strip())
        if m:
            flush_section()
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2)))
        elif not line.strip():
            flush_paragraph()
        else:
            buffer.append(line.strip())
    flush_section()
    return sections


def _pieces(paragraph: str, size: int) -> list[str]:
    words = paragraph.split()
    if len(words) <= size:
        return [paragraph]
    pieces: list[str] = []
    current: list[str] = []
    for sentence in SENTENCE.split(paragraph):
        s_words = sentence.split()
        if len(s_words) > size:  # pathological sentence: hard split on words
            if current:
                pieces.append(" ".join(current))
                current = []
            pieces += [" ".join(s_words[i : i + size]) for i in range(0, len(s_words), size)]
            continue
        if len(current) + len(s_words) > size and current:
            pieces.append(" ".join(current))
            current = []
        current += s_words
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_document(doc: Document, size: int = 180, overlap: int = 30) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    chunks: list[Chunk] = []
    for section in split_sections(doc.text):
        packed: list[list[str]] = []
        current: list[str] = []
        for paragraph in section.paragraphs:
            for piece in _pieces(paragraph, size - overlap):  # leave room for the overlap
                words = piece.split()
                if current and len(current) + len(words) > size:
                    packed.append(current)
                    keep = min(overlap, size - len(words))
                    current = current[-keep:] if keep > 0 else []
                current = current + words
        if current:
            packed.append(current)
        for words in packed:
            text = " ".join(words)
            chunks.append(
                Chunk(
                    id=f"{doc.id}:{len(chunks)}",
                    document_id=doc.id,
                    tenant=doc.tenant,
                    title=doc.title,
                    section=section.path or doc.title,
                    text=text,
                    ordinal=len(chunks),
                    source=doc.source,
                    acl=doc.acl,
                    metadata={**doc.metadata, "content_hash": hashlib.sha256(text.encode()).hexdigest()[:16]},
                )
            )
    return chunks


def embedding_text(chunk: Chunk) -> str:
    """Text that is embedded: title and section path give context to otherwise ambiguous passages."""
    return f"{chunk.title}\n{chunk.section}\n\n{chunk.text}"
