"""Cached embedder wrapper (Phase 13, Part 3).

Memoizes ``(model, text) -> vector`` behind the :class:`Embedder` interface.
Embeddings are deterministic for a fixed model, so a cache hit is always
identical to a fresh computation. The cache is bounded and shared by every
caller, so repeated embeddings (e.g. the same query embedding, or re-indexed
overlapping chunks) are served without recomputation.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.caching import EmbeddingCache
from app.embedding.base import Embedder


class CachedEmbedder(Embedder):
    """Wraps an embedder, memoizing results keyed by ``(model, text)``."""

    def __init__(self, embedder: Embedder, cache: EmbeddingCache | None = None) -> None:
        self.embedder = embedder
        # Note: `cache or EmbeddingCache()` would be a bug here — the cache
        # defines __len__, so an EMPTY cache is falsy and would be silently
        # replaced with a fresh default-sized one.
        self.cache = cache if cache is not None else EmbeddingCache()
        # The cache key must change when the underlying model changes; derive
        # it from the backend class plus its model attribute when present.
        self.model_key = (
            f"{type(embedder).__name__}:{getattr(embedder, 'model', '') or ''}"
        )

    @property
    def dimensions(self) -> int:
        return self.embedder.dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float] | None] = [None] * len(texts)
        missing: list[tuple[int, str]] = []
        for index, text in enumerate(texts):
            cached = self.cache.get(self.model_key, text)
            if cached is not None:
                vectors[index] = cached
            else:
                missing.append((index, text))
        if missing:
            fresh = self.embedder.embed([text for _, text in missing])
            for (index, text), vector in zip(missing, fresh):
                self.cache.set(self.model_key, text, vector)
                vectors[index] = vector
        return [vector for vector in vectors if vector is not None]

    def embed_one(self, text: str) -> list[float]:
        cached = self.cache.get(self.model_key, text)
        if cached is not None:
            return cached
        vector = self.embedder.embed_one(text)
        self.cache.set(self.model_key, text, vector)
        return vector

    async def close(self) -> None:
        self.cache.clear()
        await self.embedder.close()