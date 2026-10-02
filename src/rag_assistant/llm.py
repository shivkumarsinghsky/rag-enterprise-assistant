"""LLM providers.

- OpenAICompatibleChat: POST /v1/chat/completions (OpenAI, Ollama, vLLM, LiteLLM, Azure via gateway), with timeout
  and bounded retries on 429/5xx.
- ExtractiveAnswerer: offline fallback that composes an answer from the best-matching context sentences and cites
  them. It is not a language model; it exists so the pipeline, citations and evaluation run without any API key.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from rag_assistant.embeddings import tokenize
from rag_assistant.prompting import REFUSAL, Prompt

SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Generation:
    text: str
    usage: dict[str, int] = field(default_factory=dict)


class LLM(Protocol):
    name: str

    def generate(self, prompt: Prompt) -> Generation: ...

    def condense(self, history: list[dict[str, str]], question: str) -> str: ...


class OpenAICompatibleChat:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 600,
        timeout: float = 30.0,
        retries: int = 2,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport
        )

    def _chat(self, messages: list[dict[str, str]], max_tokens: int) -> Generation:
        body = {"model": self.name, "messages": messages, "temperature": self.temperature, "max_tokens": max_tokens}
        for attempt in range(self.retries + 1):
            response = self._client.post("/chat/completions", json=body)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < self.retries:
                time.sleep(min(2**attempt, 8))
                continue
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage") or {}
            return Generation(
                data["choices"][0]["message"]["content"] or "",
                {k: int(usage.get(k, 0)) for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
            )
        raise RuntimeError("unreachable")

    def generate(self, prompt: Prompt) -> Generation:
        return self._chat(prompt.messages, self.max_tokens)

    def condense(self, history: list[dict[str, str]], question: str) -> str:
        transcript = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history[-6:])
        messages = [
            {
                "role": "system",
                "content": "Rewrite the user's last question as a standalone search query using the conversation. "
                "Return only the query.",
            },
            {"role": "user", "content": f"Conversation:\n{transcript}\n\nLast question: {question}"},
        ]
        return self._chat(messages, 80).text.strip() or question


ENTITY = re.compile(r"(?<!^)(?<![.?!]\s)\b[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*")


class ExtractiveAnswerer:
    """Refuses unless (a) every named entity in the question (capitalised terms such as company names or tags)
    occurs in the context and (b) the chosen sentences cover at least `min_coverage` of the question's terms.
    These checks stand in for the judgement a language model applies with the grounding rules in the prompt."""

    name = "extractive"

    def __init__(self, max_sentences: int = 3, min_coverage: float = 0.5) -> None:
        self.max_sentences = max_sentences
        self.min_coverage = min_coverage

    def generate(self, prompt: Prompt) -> Generation:
        question = prompt.messages[-1]["content"].rsplit("Question:", 1)[-1].strip()
        context_text = " ".join(f"{sc.chunk.title} {sc.chunk.section} {sc.chunk.text}" for sc in prompt.context).lower()
        if any(e.lower() not in context_text for e in ENTITY.findall(question)):
            return Generation(REFUSAL)
        q_terms = set(tokenize(question))
        scored: list[tuple[float, int, str]] = []
        for i, sc in enumerate(prompt.context, start=1):
            for sentence in SENTENCE.split(sc.chunk.text):
                terms = set(tokenize(sentence))
                overlap = len(q_terms & terms)
                if overlap:
                    scored.append((overlap / (len(terms) ** 0.5 or 1), i, sentence.strip()))
        if not scored:
            return Generation(REFUSAL)
        best = sorted(scored, key=lambda s: -s[0])[: self.max_sentences]
        covered = set().union(*(set(tokenize(text)) for _, _, text in best)) & q_terms
        if len(covered) < self.min_coverage * len(q_terms):
            return Generation(REFUSAL)
        return Generation(" ".join(f"{text} [{i}]" for _, i, text in best))

    def condense(self, history: list[dict[str, str]], question: str) -> str:
        last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        return f"{last_user} {question}".strip() if last_user else question
