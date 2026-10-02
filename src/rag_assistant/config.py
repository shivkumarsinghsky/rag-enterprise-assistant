from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven configuration (see .env.example)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    log_level: str = "INFO"

    # --- Providers -------------------------------------------------------------------------------------
    #: "hashing" (offline, deterministic) or "openai" (any OpenAI-compatible /v1/embeddings: OpenAI, Ollama, vLLM)
    embedding_provider: str = "hashing"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=384, ge=16, le=4096)
    #: "extractive" (offline, no LLM) or "openai" (any OpenAI-compatible /v1/chat/completions)
    llm_provider: str = "extractive"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 600
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str | None = None
    request_timeout_seconds: float = 30.0

    # --- Storage ---------------------------------------------------------------------------------------
    #: "memory" or "pgvector"
    vector_store: str = "memory"
    database_url: str = "postgresql://rag:rag-local-dev@localhost:5432/rag"

    # --- Chunking and retrieval ------------------------------------------------------------------------
    chunk_size_words: int = Field(default=180, ge=20, le=2000)
    chunk_overlap_words: int = Field(default=30, ge=0)
    top_k: int = Field(default=5, ge=1, le=50)
    candidate_k: int = Field(default=20, ge=1, le=200)
    min_relevance: float = 0.05
    context_budget_words: int = Field(default=1200, ge=100)

    # --- Security --------------------------------------------------------------------------------------
    #: JSON: {"<api key>": {"tenant": "acme", "user": "alice", "groups": ["engineering"]}}
    api_keys: str = '{"dev-acme-key": {"tenant": "acme", "user": "dev", "groups": ["all-staff", "maintenance"]}}'


@lru_cache
def get_settings() -> Settings:
    return Settings()
