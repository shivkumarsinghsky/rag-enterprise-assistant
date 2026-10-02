from pathlib import Path

import pytest

from rag_assistant.__main__ import ingest_directory
from rag_assistant.config import Settings
from rag_assistant.embeddings import HashingEmbedder
from rag_assistant.llm import ExtractiveAnswerer
from rag_assistant.models import Principal
from rag_assistant.pipeline import RagPipeline
from rag_assistant.vectorstore.memory import InMemoryVectorStore

ROOT = Path(__file__).resolve().parents[1]
MAINTENANCE = Principal("acme", "alice", frozenset({"maintenance", "all-staff"}))
STAFF = Principal("acme", "bob", frozenset({"all-staff"}))
EXEC = Principal("acme", "carol", frozenset({"executives"}))
GLOBEX = Principal("globex", "gina", frozenset())


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.fixture
def pipeline(settings: Settings) -> RagPipeline:
    p = RagPipeline(settings, InMemoryVectorStore(), HashingEmbedder(384), ExtractiveAnswerer())
    ingest_directory(p, ROOT / "sample_docs")
    return p
