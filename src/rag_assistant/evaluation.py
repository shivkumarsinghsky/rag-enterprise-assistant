"""Offline evaluation of retrieval and answers against a labelled question set (JSONL).

Each line: {"question": "...", "tenant": "acme", "groups": ["maintenance"],
            "expected_documents": ["pump-p101-manual"], "must_include": ["90 days"], "answerable": true}

Metrics:
- hit_rate@k        expected document appears in the retrieved top-k
- mrr               mean reciprocal rank of the first relevant document
- citation_precision share of citations that point to an expected document
- answer_recall     share of `must_include` phrases present in the answer
- refusal_accuracy  unanswerable questions (e.g. other tenant's data, no source) are refused
Run in CI to catch regressions when chunking, embeddings, prompts or models change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rag_assistant.models import Principal
from rag_assistant.pipeline import RagPipeline


@dataclass
class EvalReport:
    questions: int
    hit_rate: float
    mrr: float
    citation_precision: float
    answer_recall: float
    refusal_accuracy: float
    failures: list[str]

    def as_dict(self) -> dict[str, object]:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def evaluate(pipeline: RagPipeline, cases_path: Path) -> EvalReport:
    cases = [json.loads(line) for line in cases_path.read_text().splitlines() if line.strip()]
    hits = rr = cit_ok = cit_total = recall_sum = 0.0
    answerable = refusals_ok = unanswerable = 0
    failures: list[str] = []
    for case in cases:
        principal = Principal(case["tenant"], "eval", frozenset(case.get("groups", [])))
        answer = pipeline.ask(principal, case["question"])
        expected = set(case.get("expected_documents", []))
        if not case.get("answerable", True):
            unanswerable += 1
            if not answer.grounded:
                refusals_ok += 1
            else:
                failures.append(f"should refuse: {case['question']}")
            continue
        answerable += 1
        ranked_docs = [sc.chunk.document_id for sc in answer.retrieved]
        rank = next((i for i, d in enumerate(ranked_docs, start=1) if d in expected), None)
        if rank:
            hits += 1
            rr += 1 / rank
        else:
            failures.append(f"not retrieved: {case['question']}")
        cit_total += len(answer.citations)
        cit_ok += sum(1 for c in answer.citations if c.document_id in expected)
        phrases = case.get("must_include", [])
        found = sum(1 for p in phrases if p.lower() in answer.text.lower())
        recall_sum += found / len(phrases) if phrases else 1.0
        if phrases and found < len(phrases):
            failures.append(
                f"missing {[p for p in phrases if p.lower() not in answer.text.lower()]}: {case['question']}"
            )
    n = answerable or 1
    return EvalReport(
        questions=len(cases),
        hit_rate=hits / n,
        mrr=rr / n,
        citation_precision=cit_ok / cit_total if cit_total else 0.0,
        answer_recall=recall_sum / n,
        refusal_accuracy=refusals_ok / unanswerable if unanswerable else 1.0,
        failures=failures,
    )
