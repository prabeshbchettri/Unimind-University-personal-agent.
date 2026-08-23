"""Cached retriever wrapper (Phase 13, Part 3).

Decorates any object implementing ``async retrieve(query, top_k)`` (the
retrievers and the adaptive router) with the :class:`RetrievalCache`.
Results are serialized to plain dicts on store and rebuilt on hit.

Staleness is impossible by construction: the indexing service calls
``cache.invalidate()`` after every document mutation, so an entry is served
only while the underlying index is byte-identical to what produced it.
"""

from __future__ import annotations

from dataclasses import asdict

from app.caching import RetrievalCache
from app.retrieval.models import SearchResult


class CachedRetriever:
    """Same interface as the retrievers, with caching in front."""

    def __init__(
        self,
        retriever,
        cache: RetrievalCache | None = None,
        name: str = "",
    ) -> None:
        self.retriever = retriever
        self.cache = cache
        self.name = name or type(retriever).__name__

    async def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return cached results when current, otherwise retrieve and store."""
        if self.cache is None:
            return await self.retriever.retrieve(query, top_k=top_k)

        hit = self.cache.get(self.name, query, top_k)
        if hit is not None:
            return [SearchResult(**item) for item in hit]

        results = await self.retriever.retrieve(query, top_k=top_k)
        self.cache.set(
            self.name,
            query,
            top_k,
            [asdict(result) for result in results],
        )
        return results