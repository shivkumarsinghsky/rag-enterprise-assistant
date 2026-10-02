"""Input and context guardrails.

Retrieved documents are *data*, not instructions. Defences, in order of importance:
1. Authorization is enforced in retrieval (tenant + ACL filters), so the model never sees content the user cannot.
2. Context is delimited and the system prompt says instructions inside it must be ignored.
3. Chunks that look like prompt-injection attempts are excluded and reported (this module).
4. Output is post-checked: citations must reference provided context (see citations.py).
Heuristics are not a complete defence; they reduce risk and make attempts visible.
"""

from __future__ import annotations

import re

INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts?|rules)",
        r"\bdisregard\s+(all\s+|the\s+)?(previous|prior|above|system)\b",
        r"\byou\s+are\s+now\s+(a|an|in)\b",
        r"\b(reveal|print|show|output)\s+(the\s+|your\s+)?(system\s+prompt|hidden\s+instructions|api\s+key|password)",
        r"<\s*/?\s*(system|assistant)\s*>",
        r"\bnew\s+instructions\s*:",
    )
]

PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[email]"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[card]"),
    (re.compile(r"\+?\d{1,3}[ -]?\(?\d{2,4}\)?[ -]?\d{3,4}[ -]?\d{3,4}\b"), "[phone]"),
]

MAX_QUESTION_CHARS = 2_000


class GuardrailViolation(ValueError):
    pass


def looks_like_injection(text: str) -> bool:
    return any(p.search(text) for p in INJECTION_PATTERNS)


def check_question(question: str) -> str:
    q = question.strip()
    if not q:
        raise GuardrailViolation("question is empty")
    if len(q) > MAX_QUESTION_CHARS:
        raise GuardrailViolation(f"question exceeds {MAX_QUESTION_CHARS} characters")
    return q


def redact_pii(text: str) -> str:
    """For logs and traces: never store raw personal data in observability systems."""
    for pattern, replacement in PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text
