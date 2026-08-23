"""Qdrant repository.

Vector storage/retrieval over Qdrant. Supports an in-memory client (no server,
used by tests and offline dev), a local path client, and a remote HTTP client.

Qdrant holds *only* vector search data (chunks + payload). Chat history lives
in PostgreSQL (Phase 10) and knowledge-graph relationships in Neo4j (Phase 7).
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections.abc import Sequence

from qdrant_client import QdrantClient, models

from app.repositories.base import Repository
from app.retrieval.models import SearchHit

_IN_MEMORY_URLS = {"", ":memory:", "in-memory", "memory"}


def point_uuid(chunk_id: str) -> uuid.UUID:
    """Stable, deterministic UUID for a chunk id (idempotent upserts)."""
    return uuid.UUID(hex=hashlib.md5(chunk_id.encode("utf-8")).hexdigest())


class QdrantRepository(Repository):
    """Vector store repository backed by Qdrant."""

    def __init__(
        self,
        url: str = ":memory:",
        api_key: str = "",
        collection_names: Sequence[str] = (),
        default_dimensions: int = 384,
    ) -> None:
        super().__init__()
        self._collection_names = list(collection_names) or ["university_docs", "past_questions", "library_books"]
        self._default_dimensions = default_dimensions
        self._client = self._build_client(url, api_key)

    @property
    def collection_names(self) -> list[str]:
        return self._collection_names

    def _build_client(self, url: str, api_key: str) -> QdrantClient:
        if url in _IN_MEMORY_URLS:
            return QdrantClient(":memory:")
        return QdrantClient(url=url, api_key=api_key or None, timeout=10)

    async def ping(self) -> bool:
        def _ping() -> bool:
            try:
                self._client.get_collections()
                return True
            except Exception:
                return False

        return await asyncio.to_thread(_ping)

    async def ensure_collections(self, dimensions: int | None = None) -> None:
        """Create any missing collections with the given vector size."""
        dims = dimensions or self._default_dimensions

        def _ensure() -> None:
            for name in self._collection_names:
                if not self._client.collection_exists(name):
                    self._client.create_collection(
                        collection_name=name,
                        vectors_config=models.VectorParams(
                            size=dims,
                            distance=models.Distance.COSINE,
                        ),
                    )

        await asyncio.to_thread(_ensure)

    async def collection_exists(self, name: str) -> bool:
        return await asyncio.to_thread(self._client.collection_exists, name)

    async def upsert_points(
        self,
        collection: str,
        points: Sequence[dict],
    ) -> None:
        """Upsert ``[{"id", "vector", "payload"}]`` dicts into ``collection``."""

        def _upsert() -> None:
            structs = [
                models.PointStruct(
                    id=str(point["id"]),
                    vector=point["vector"],
                    payload=point.get("payload", {}),
                )
                for point in points
            ]
            self._client.upsert(collection, structs)

        await asyncio.to_thread(_upsert)

    async def search(
        self,
        collection: str,
        vector: Sequence[float],
        top_k: int = 5,
    ) -> list[SearchHit]:
        """Return the ``top_k`` nearest points to ``vector`` in ``collection``."""

        def _search() -> list[SearchHit]:
            hits = self._client.query_points(
                collection_name=collection,
                query=list(vector),
                limit=top_k,
                with_payload=True,
            ).points
            return [
                SearchHit(
                    chunk_id=str(hit.id),
                    score=float(hit.score),
                    payload=hit.payload or {},
                )
                for hit in hits
            ]

        return await asyncio.to_thread(_search)

    async def count(self, collection: str) -> int:
        return (await asyncio.to_thread(self._client.count, collection)).count

    async def list_points(self, collection: str, limit: int = 5000) -> list[dict]:
        """Return the payloads of all points in ``collection`` (no vectors).

        Used by the recommendation service (Phase 9) to build book profiles
        from the indexed chunk payloads without loading vectors.
        """
        payloads: list[dict] = []
        offset: object = None

        def _scroll(page_offset: object | None):
            points, next_offset = self._client.scroll(
                collection_name=collection,
                limit=min(limit, 1000),
                offset=page_offset,
                with_payload=True,
                with_vectors=False,
            )
            return [point.payload or {} for point in points], next_offset

        while True:
            page, offset = await asyncio.to_thread(_scroll, offset)
            payloads.extend(page)
            if offset is None or len(payloads) >= limit:
                break
        return payloads

    async def close(self) -> None:
        await asyncio.to_thread(self._client.close)