"""Qdrant vector store integration.

Responsibilities: collection initialization, deterministic point IDs, upsert of
chunks with metadata payloads, and similarity search. Everything else in the
system depends on this small class, not on the ``qdrant_client`` types, so the
storage backend stays replaceable and tests can substitute a fake.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from app.ingestion import Chunk

#: Directory used by the embedded local engine when ``QDRANT_URL=local``.
LOCAL_STORAGE_DIR = Path(__file__).resolve().parents[2] / "data" / "qdrant_storage"


def _build_client(url: str) -> QdrantClient:
    """Create a Qdrant client from the configured URL.

    Supported forms:

    - ``local``: Qdrant's real engine embedded in this process, persisted to
      ``data/qdrant_storage`` (the default, server-free development setup)
    - ``:memory:``: embedded engine, ephemeral (used by tests)
    - ``path:<dir>``: embedded engine persisted to a specific directory
    - ``http://...``: a Qdrant server (not required for local development)
    """
    if url == "local":
        LOCAL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(LOCAL_STORAGE_DIR))
    if url.startswith("path:"):
        return QdrantClient(path=url.removeprefix("path:"))
    if url == ":memory:":
        return QdrantClient(":memory:")
    return QdrantClient(url=url, timeout=10, check_compatibility=False)


class VectorStoreError(RuntimeError):
    """Raised when the vector store cannot be initialized or queried."""


#: Payload keys stored with every point. ``text`` is kept in the payload so
#: retrieved chunks can later be assembled into LLM context.
PAYLOAD_KEYS = ("chunk_id", "document_name", "source_path", "page", "chunk_index", "text")


@dataclass(frozen=True)
class SearchResult:
    """One similarity-search hit with its source metadata and score."""

    score: float
    chunk_id: str
    document_name: str
    source_path: str
    page: int
    chunk_index: int
    text: str


def _point_id(chunk_id: str) -> str:
    """Map a stable chunk id to a stable Qdrant point id.

    Deterministic (UUIDv5) so re-ingesting the same document overwrites its old
    points instead of duplicating them.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"unirag:{chunk_id}"))


def _chunk_payload(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.id,
        "document_name": chunk.document_name,
        "source_path": chunk.source_path,
        "page": chunk.page,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
    }


class QdrantVectorStore:
    """Thin, explicit wrapper around the Qdrant client for this project's needs."""

    def __init__(self, url: str, collection: str, dim: int, client: QdrantClient | None = None) -> None:
        if dim <= 0:
            raise VectorStoreError(f"Vector dimension must be positive, got {dim}")
        self._collection = collection
        self._dim = dim
        self._client = client if client is not None else _build_client(url)

    def ensure_collection(self) -> bool:
        """Create the collection if needed. Returns True if it was created.

        If the collection already exists, its configured vector dimension is
        verified against ``self._dim`` so a mismatch fails loudly instead of
        producing silent upsert errors later.
        """
        if self._client.collection_exists(self._collection):
            info = self._client.get_collection(self._collection)
            try:
                existing_dim = int(info.config.params.vectors.size)
            except (AttributeError, TypeError, ValueError) as exc:
                raise VectorStoreError(
                    f"Collection '{self._collection}' exists but its vector size "
                    "cannot be verified; check it manually"
                ) from exc
            if existing_dim != self._dim:
                raise VectorStoreError(
                    f"Collection '{self._collection}' has vector size {existing_dim} "
                    f"but the embedding client produces {self._dim}. "
                    "Use a different collection name or recreate the collection."
                )
            return False
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(
                size=self._dim, distance=models.Distance.COSINE
            ),
        )
        return True

    def ping(self) -> None:
        """Raise ``VectorStoreError`` if the store cannot answer a cheap call.

        Used by the ``/ready`` endpoint. Deliberately catches every exception:
        for a liveness probe any failure means "not ready", and the cause is
        preserved in the error message.
        """
        try:
            self._client.get_collection(self._collection)
        except Exception as exc:  # noqa: BLE001 -- probe must classify, not crash
            raise VectorStoreError(f"Vector store is not ready: {exc}") from exc

    def reset(self) -> None:
        """Delete the collection if it exists (used by `--recreate`)."""
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        """Upsert chunks with their embedding vectors. Returns the point count."""
        if len(chunks) != len(vectors):
            raise VectorStoreError(
                f"Got {len(chunks)} chunks but {len(vectors)} vectors"
            )
        if not chunks:
            return 0
        points = [
            models.PointStruct(
                id=_point_id(chunk.id),
                vector=vector,
                payload=_chunk_payload(chunk),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        try:
            self._client.upsert(collection_name=self._collection, points=points, wait=True)
        except Exception as exc:
            raise VectorStoreError(f"Failed to upsert points: {exc}") from exc
        return len(points)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[SearchResult]:
        """Return the ``top_k`` most similar chunks with metadata and scores."""
        if top_k <= 0:
            raise VectorStoreError(f"top_k must be positive, got {top_k}")
        try:
            response = self._client.query_points(
                collection_name=self._collection,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"Similarity search failed: {exc}") from exc

        results: list[SearchResult] = []
        for point in response.points:
            payload = point.payload or {}
            missing = [key for key in ("chunk_id", "document_name", "text") if key not in payload]
            if missing:
                raise VectorStoreError(
                    f"Point payload is missing required keys: {', '.join(missing)}"
                )
            results.append(
                SearchResult(
                    score=float(point.score),
                    chunk_id=str(payload.get("chunk_id", "")),
                    document_name=str(payload.get("document_name", "")),
                    source_path=str(payload.get("source_path", "")),
                    page=int(payload.get("page", 0)),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    text=str(payload.get("text", "")),
                )
            )
        return results

    def scroll_chunks(self, limit: int = 50_000) -> list[Chunk]:
        """Return every stored chunk (reconstructed from payloads).

        Used to build the BM25 index for HYBRID retrieval: Qdrant payloads are
        the single source of truth for the corpus, so the lexical index always
        matches the vector store without a second storage system.
        """
        try:
            points, _ = self._client.scroll(
                collection_name=self._collection,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to read collection: {exc}") from exc

        chunks: list[Chunk] = []
        for point in points:
            payload = point.payload or {}
            missing = [key for key in PAYLOAD_KEYS if key not in payload]
            if missing:
                raise VectorStoreError(
                    f"Point payload is missing required keys: {', '.join(missing)}"
                )
            chunks.append(
                Chunk(
                    id=str(payload["chunk_id"]),
                    document_name=str(payload["document_name"]),
                    source_path=str(payload["source_path"]),
                    page=int(payload["page"]),
                    chunk_index=int(payload["chunk_index"]),
                    text=str(payload["text"]),
                )
            )
        return chunks
