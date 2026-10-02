"""Embedding providers.

- HashingEmbedder: deterministic feature-hashing of word unigrams and bigrams into a fixed-size, L2-normalised
  vector. No model download or network — used for tests, CI and offline demos. It captures lexical overlap only,
  not semantics; use a real embedding model for meaningful semantic retrieval.
- OpenAICompatibleEmbedder: any service implementing POST /v1/embeddings (OpenAI, Azure OpenAI via a gateway,
  Ollama, vLLM, LiteLLM).
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import httpx
import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.float32]
TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
STOPWORDS = frozenset(
    "a an and are as at be by can could do does during for from get has have how i if in is it its "
    "many may might much must need "
    "of on only or our should that the their there this to was we what when where which who why will with "
    "use used would you your".split()
)
SUFFIXES = ("ment", "ing", "ion", "ed", "es", "s", "e")


def stem(token: str) -> str:
    """Very light suffix stripping so "inspection"/"inspect" and "removed"/"remove" match. Not a full stemmer."""
    for _ in range(2):  # e.g. replacement -> replace -> replac
        for suffix in SUFFIXES:
            if len(token) > len(suffix) + 3 and token.endswith(suffix) and not token[-len(suffix) - 1].isdigit():
                token = token[: -len(suffix)]
                break
        else:
            break
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class Embedder(Protocol):
    dimensions: int
    model_name: str

    def embed(self, texts: list[str]) -> list[Vector]: ...


class HashingEmbedder:
    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions
        self.model_name = f"hashing-{dimensions}"

    def _index(self, feature: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
        return h % self.dimensions, 1.0 if (h >> 63) & 1 else -1.0

    def embed(self, texts: list[str]) -> list[Vector]:
        out: list[Vector] = []
        for text in texts:
            v = np.zeros(self.dimensions, dtype=np.float32)
            tokens = tokenize(text)
            features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
            for f in features:
                i, sign = self._index(f)
                v[i] += sign
            norm = float(np.linalg.norm(v))
            out.append(v / norm if norm else v)
        return out


class OpenAICompatibleEmbedder:
    def __init__(self, base_url: str, api_key: str | None, model: str, dimensions: int, timeout: float = 30.0) -> None:
        self.model_name = model
        self.dimensions = dimensions
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)

    def embed(self, texts: list[str]) -> list[Vector]:
        out: list[Vector] = []
        for start in range(0, len(texts), 64):  # batch to stay under provider request limits
            batch = texts[start : start + 64]
            response = self._client.post("/embeddings", json={"model": self.model_name, "input": batch})
            response.raise_for_status()
            data = sorted(response.json()["data"], key=lambda d: d["index"])
            for item in data:
                v = np.asarray(item["embedding"], dtype=np.float32)
                if v.shape[0] != self.dimensions:
                    raise ValueError(f"embedding has {v.shape[0]} dimensions, expected {self.dimensions}")
                out.append((v / (np.linalg.norm(v) or 1.0)).astype(np.float32))
        return out
