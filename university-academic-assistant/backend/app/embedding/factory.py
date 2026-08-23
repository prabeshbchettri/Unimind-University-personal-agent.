"""Embedder factory.

Builds an :class:`Embedder` from settings:

- ``deterministic`` — local token-hash vectors (tests / offline dev)
- ``ollama`` — BGE-style embeddings via Ollama ``/api/embed``
- ``auto`` — Ollama when a base URL is configured, otherwise deterministic
"""

from __future__ import annotations

from app.caching import EmbeddingCache
from app.config import Settings
from app.embedding.base import Embedder
from app.embedding.cached import CachedEmbedder
from app.embedding.deterministic import DeterministicEmbedder
from app.embedding.ollama import OllamaEmbedder


def build_embedder(settings: Settings) -> Embedder:
    """Return the configured embedder backend, memoized per text."""
    backend = (settings.embedder_backend or "auto").strip().lower()

    if backend == "deterministic":
        embedder: Embedder = DeterministicEmbedder(dimensions=settings.embedding_dimensions)
    elif backend == "ollama":
        embedder = OllamaEmbedder(
            base_url=settings.ollama_base_url,
            model=settings.embedding_model,
            default_dimensions=settings.embedding_dimensions,
        )
    elif settings.ollama_base_url:
        embedder = OllamaEmbedder(
            base_url=settings.ollama_base_url,
            model=settings.embedding_model,
            default_dimensions=settings.embedding_dimensions,
        )
    else:
        embedder = DeterministicEmbedder(dimensions=settings.embedding_dimensions)

    return CachedEmbedder(
        embedder,
        EmbeddingCache(max_entries=settings.embedding_cache_max_entries),
    )
