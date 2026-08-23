"""Vector indexing service.

Phase 4: orchestrates the pipeline

    StructuredDocument -> SemanticChunker -> Embedder -> Qdrant

and provides vector search over the indexed collections. This is retrieval
only — answer generation (RAG) arrives in Phase 5.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Sequence

from app.chunking import SemanticChunker
from app.config import settings
from app.embedding import Embedder, build_embedder
from app.repositories.graph import GraphUnavailableError
from app.retrieval import SearchResult, all_collections, collection_for
from app.retrieval.sparse import BM25Index
from app.repositories.qdrant import QdrantRepository, point_uuid
from app.services.base import Service
from app.structure.models import StructuredDocument


class VectorIndexingService(Service):
    """Chunks, embeds and indexes structured documents, and searches them."""

    def __init__(
        self,
        chunker: SemanticChunker | None = None,
        embedder: Embedder | None = None,
        repository: QdrantRepository | None = None,
        collection_names: Sequence[str] | None = None,
        sparse_index: BM25Index | None = None,
        graph_service=None,
        retrieval_cache=None,
    ) -> None:
        super().__init__()
        self.chunker = chunker or SemanticChunker(
            max_chars=settings.chunk_max_chars,
            overlap_chars=settings.chunk_overlap_chars,
        )
        self.embedder = embedder or build_embedder(settings)
        self.collection_names = list(collection_names or settings.collection_names)
        self.repository = repository or QdrantRepository(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_names=self.collection_names,
            default_dimensions=settings.embedding_dimensions,
        )
        # Optional BM25 keyword index (Phase 6): every indexed chunk is also
        # tokenized here so hybrid retrieval can match exact terms.
        self.sparse_index = sparse_index
        # Optional knowledge graph (Phase 7): the same document is also written
        # to the academic entity graph (explicit relationships, not vectors).
        self.graph_service = graph_service
        # Optional retrieval cache (Phase 13): invalidated after every
        # mutation so cached retrieval can never serve stale data.
        self.retrieval_cache = retrieval_cache

    async def health(self) -> dict:
        try:
            reachable = await self.repository.ping()
        except Exception as exc:  # pragma: no cover - defensive
            return {"name": self.name, "status": "error", "error": str(exc)}
        return {
            "name": self.name,
            "status": "ok" if reachable else "unavailable",
            "embedder": type(self.embedder).__name__,
            "collections": self.collection_names,
        }

    async def index(self, structured: StructuredDocument) -> dict:
        """Chunk, embed and store ``structured``. Returns an index summary."""
        chunks = await asyncio.to_thread(self.chunker.chunk, structured)
        if not chunks:
            return {
                "document_id": structured.document_id,
                "chunk_count": 0,
                "collections": {},
            }

        texts = [chunk.text for chunk in chunks]
        vectors = await asyncio.to_thread(self.embedder.embed, texts)
        await self.repository.ensure_collections(self.embedder.dimensions)

        grouped: dict[str, list] = defaultdict(list)
        for chunk, vector in zip(chunks, vectors):
            grouped[collection_for(chunk.document_type, self.collection_names)].append(
                (chunk, vector)
            )

        counts: dict[str, int] = {}
        # Library books: also carry the book's structural subtopics on every
        # chunk so recommendation profiles can see subsection coverage even
        # when a chapter's blocks merged into one chunk.
        book_subtopics: list[str] = []
        if structured.metadata.document_type == "library_book":
            seen: set[str] = set()
            for chapter in structured.structure.chapters or ():
                for subtopic in chapter.subtopics or ():
                    if subtopic and subtopic not in seen:
                        seen.add(subtopic)
                        book_subtopics.append(subtopic)
        for collection, items in grouped.items():
            points = [
                {
                    "id": point_uuid(chunk.chunk_id),
                    "vector": vector,
                    "payload": (
                        {**chunk.payload(), "subtopics": book_subtopics}
                        if book_subtopics
                        else chunk.payload()
                    ),
                }
                for chunk, vector in items
            ]
            await self.repository.upsert_points(collection, points)
            counts[collection] = len(items)

        if self.sparse_index is not None:
            # Keep the BM25 keyword index in sync with the vector index. All
            # chunks of one structured document share a collection.
            for collection, items in grouped.items():
                self.sparse_index.add_chunks([chunk for chunk, _ in items], collection)

        if self.retrieval_cache is not None:
            # The index changed: any cached retrieval is now stale by
            # construction (no stale data can be served after ingestion).
            self.retrieval_cache.invalidate()

        graph_summary = None
        if self.graph_service is not None:
            try:
                graph_summary = await self.graph_service.index(structured)
            except GraphUnavailableError as exc:
                # The graph is a secondary index; an unreachable graph must not
                # take down vector indexing.
                self.logger.warning("Skipped graph indexing", extra={"extra_fields": {"error": str(exc)}})
                graph_summary = {"status": "unavailable", "error": str(exc)}

        return {
            "document_id": structured.document_id,
            "chunk_count": len(chunks),
            "collections": counts,
            "graph": graph_summary,
        }

    async def search(
        self,
        query: str,
        collection: str | None = None,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Vector search over one collection, or all collections by default.

        Returns results with ``text``, ``score`` and ``metadata``.
        """
        if not query.strip():
            return []

        vector = (await asyncio.to_thread(self.embedder.embed, [query]))[0]
        collections = [collection] if collection else all_collections(self.collection_names)

        results: list[SearchResult] = []
        for name in collections:
            if not await self.repository.collection_exists(name):
                continue
            hits = await self.repository.search(name, vector, top_k)
            for hit in hits:
                payload = hit.payload or {}
                results.append(
                    SearchResult(
                        text=payload.get("text", ""),
                        score=hit.score,
                        metadata={key: value for key, value in payload.items() if key != "text"},
                        collection=name,
                    )
                )

        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]