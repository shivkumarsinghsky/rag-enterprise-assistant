"""Okapi BM25 keyword scoring — catches exact terms (part numbers, error codes, acronyms) that embeddings blur."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from rag_assistant.embeddings import tokenize


def bm25_scores(query: str, documents: Sequence[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    q_terms = tokenize(query)
    if not documents or not q_terms:
        return [0.0] * len(documents)
    tokenized = [tokenize(d) for d in documents]
    avg_len = sum(len(t) for t in tokenized) / len(tokenized) or 1.0
    df: Counter[str] = Counter()
    for tokens in tokenized:
        df.update(set(tokens))
    n = len(documents)
    scores = []
    for tokens in tokenized:
        tf = Counter(tokens)
        score = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * len(tokens) / avg_len))
        scores.append(score)
    return scores
