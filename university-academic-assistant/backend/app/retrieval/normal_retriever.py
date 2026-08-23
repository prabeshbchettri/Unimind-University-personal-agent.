"""Normal retriever (Phase 5).

The first retrieval strategy: given a user query, embed it, search the Qdrant
collections, and return the top-K chunks with their text, metadata and scores.

Phase 5 has no query router, so the retriever searches across all collections
(`university_docs`, `past_questions`, `library_books`) and merges the results
by similarity score. Hybrid (BM25) and graph retrieval arrive in later phases.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING

from app.config import settings
from app.embedding import Embedder
from app.retrieval.collections import DEFAULT_COLLECTIONS, all_collections
from app.retrieval.models import SearchResult

if TYPE_CHECKING:
    from app.repositories.qdrant import QdrantRepository


class NormalRetriever:
    """Embeds a query and retrieves the closest indexed chunks."""

    def __init__(
        self,
        embedder: Embedder,
        repository: QdrantRepository,
        collection_names: Sequence[str] | None = None,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> None:
        self.embedder = embedder
        self.repository = repository
        self.collection_names = list(collection_names) or list(DEFAULT_COLLECTIONS)
        self.top_k = top_k
        # Relevance floor: chunks scoring below this are treated as no evidence
        # (0.0 disables gating). Set to a small positive value in environments
        # where weak matches must be ignored.
        self.min_score = min_score

    async def retrieve(
        self,
        query: str,
        collection: str | None = None,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return the top ``top_k`` chunks for ``query``.

        ``collection`` restricts the search to a single collection; by default
        all collections are searched and merged. Results carry ``text``,
        ``score`` and ``metadata``.
        """
        limit = top_k or self.top_k
        if not query.strip() or limit <= 0:
            return []

        vector = await asyncio.to_thread(self.embedder.embed_one, query)
        collections = [collection] if collection else all_collections(self.collection_names)

        results: list[SearchResult] = []
        for name in collections:
            if not await self.repository.collection_exists(name):
                continue
            hits = await self.repository.search(name, vector, limit)
            for hit in hits:
                payload = hit.payload or {}
                results.append(
                    SearchResult(
                        text=payload.get("text", ""),
                        score=hit.score,
                        metadata={key: value for key, value in payload.items() if key != "text"},
                        collection=name,
                        method="dense",
                    )
                )

        results.sort(key=lambda result: result.score, reverse=True)
        results = [result for result in results if result.score >= self.min_score]
        return results[:limit]
