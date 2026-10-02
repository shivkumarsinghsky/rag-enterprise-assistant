"""Post-generation checks: keep only citations that point at provided context."""

from __future__ import annotations

import re

from rag_assistant.models import Citation, ScoredChunk
from rag_assistant.prompting import REFUSAL

MARKER = re.compile(r"\[(\d{1,3})\]")


def resolve_citations(text: str, context: list[ScoredChunk]) -> tuple[str, list[Citation], bool]:
    """Returns (cleaned text, citations in order of first use, grounded flag).

    Markers that reference a non-existent source (hallucinated citations) are removed from the text.
    An answer is *grounded* when it is not a refusal and cites at least one valid source.
    """
    citations: list[Citation] = []
    seen: set[int] = set()

    def keep(match: re.Match[str]) -> str:
        n = int(match.group(1))
        if not 1 <= n <= len(context):
            return ""
        if n not in seen:
            seen.add(n)
            c = context[n - 1].chunk
            citations.append(Citation(n, c.id, c.document_id, c.title, c.section, c.source))
        return match.group(0)

    cleaned = re.sub(r"\s{2,}", " ", MARKER.sub(keep, text)).strip()
    grounded = bool(citations) and REFUSAL.lower() not in cleaned.lower()
    return cleaned, citations, grounded
