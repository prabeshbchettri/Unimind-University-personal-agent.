"""Centralized application configuration.

Every deployment-specific value (ports, URLs, models, API keys) is read from
environment variables, optionally seeded from a local ``.env`` file. Nothing
secret is hardcoded in the source.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Every deployment-specific value (ports, URLs, models, API keys) is read
    from environment variables, optionally seeded from a local ``.env`` file.
    Nothing secret is hardcoded in the source.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Adaptive University RAG Assistant"
    app_env: str = "development"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    # Origins allowed to call the API from a browser (comma-separated). The
    # Vite dev server default first; tighten for real deployments.
    cors_allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Vector store. "local" uses Qdrant's embedded engine persisted to
    # data/qdrant_storage; a URL such as http://localhost:6333 targets a server.
    qdrant_url: str = "local"
    qdrant_collection: str = "university_docs"
    embedding_dim: int = 384

    # Document ingestion
    documents_dir: str = "../data/documents"
    chunk_size: int = 900
    chunk_overlap: int = 150

    # Embeddings ("local" uses fastembed, "ollama" uses a local Ollama server)
    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    ollama_embedding_model: str = "nomic-embed-text"

    # Retrieval and context construction
    retrieval_top_k: int = 5
    # Chunks below this similarity score are treated as insufficient evidence.
    min_relevance_score: float = 0.45
    # HYBRID-only gate for chunks found lexically but not in the vector top-k:
    # they pass only when their BM25 score reaches this threshold.
    min_bm25_score: float = 1.0
    max_context_chars: int = 4000

    # LLM gateway: "auto" (Groq primary -> Ollama fallback), "groq" or "ollama"
    llm_provider: str = "auto"
    llm_timeout: float = 120.0
    llm_temperature: float = 0.1

    # Primary hosted provider: Groq.
    groq_api_key: str = ""  # required for "groq"; "auto" skips Groq when absent
    # Groq decommissioned the llama-3.x lineup (2026): verified live via /models.
    # gpt-oss-20b is the cheapest active model with large context + structured outputs.
    groq_model: str = "openai/gpt-oss-20b"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # Local provider / fallback: Ollama.
    ollama_base_url: str = "http://localhost:11434"
    # llama3.1 (8B) is the smallest locally tested model that stays grounded
    # with full-size contexts; 3B models degenerate on long prompts.
    ollama_model: str = "llama3.1"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance.

    Cached so the ``.env`` file is read once per process. Tests can call
    ``get_settings.cache_clear()`` to force a re-read.
    """
    return Settings()
