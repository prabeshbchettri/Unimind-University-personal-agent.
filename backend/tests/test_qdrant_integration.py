"""Integration verification against the real Qdrant engine.

Two layers:

1. ``test_qdrant_local_engine_round_trip`` runs against Qdrant's real engine in
   in-process local mode (``QdrantClient(":memory:")``). It always runs and is
   never faked - it exercises actual Qdrant collection creation, upsert and
   similarity search.
2. ``test_live_qdrant_server_round_trip`` runs against a dedicated Qdrant
   *server* (``QDRANT_URL`` pointing at ``http://...``). It is skipped unless a
   server is actually reachable, so the suite passes without one.
"""

from __future__ import annotations

import uuid

import pytest
from qdrant_client import QdrantClient

from app.config import get_settings
from app.ingestion import Chunk
from app.vectorstore import QdrantVectorStore

DIM = 384


def _vector(index: int) -> list[float]:
    """One-hot-ish unit vectors that are clearly distinguishable."""
    vector = [0.0] * DIM
    vector[index % DIM] = 1.0
    return vector


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            id=f"attendance_policy.pdf::p1::c{i}",
            document_name="attendance_policy.pdf",
            source_path="data/documents/attendance_policy.pdf",
            page=1,
            chunk_index=i,
            text=text,
        )
        for i, text in enumerate(
            [
                "minimum attendance requirement is seventy five percent",
                "examination eligibility requires good standing",
            ]
        )
    ]


def test_qdrant_local_engine_round_trip() -> None:
    """Real Qdrant engine (in-process): create -> upsert -> search -> metadata."""
    store = QdrantVectorStore(
        url=":memory:",
        collection=f"it_{uuid.uuid4().hex[:8]}",
        dim=DIM,
        client=QdrantClient(":memory:"),
    )

    assert store.ensure_collection() is True
    assert store.ensure_collection() is False  # now exists, dimension verified

    chunks = _chunks()
    vectors = [_vector(0), _vector(1)]
    assert store.upsert_chunks(chunks, vectors) == 2

    hits = store.search(query_vector=_vector(0), top_k=2)
    assert len(hits) == 2
    assert hits[0].chunk_id == "attendance_policy.pdf::p1::c0"
    assert hits[0].document_name == "attendance_policy.pdf"
    assert hits[0].source_path == "data/documents/attendance_policy.pdf"
    assert hits[0].page == 1
    assert hits[0].chunk_index == 0
    assert "attendance" in hits[0].text
    assert hits[0].score > hits[1].score

    # A reset removes everything...
    store.reset()
    assert store.ensure_collection() is True
    assert store.search(query_vector=_vector(0), top_k=2) == []


def _qdrant_server_reachable(url: str) -> bool:
    try:
        client = QdrantClient(url=url, timeout=2, check_compatibility=False)
        client.get_collections()
    except Exception:
        return False
    return True


@pytest.mark.skipif(
    not _qdrant_server_reachable(get_settings().qdrant_url),
    reason=f"no Qdrant server reachable at {get_settings().qdrant_url}",
)
def test_live_qdrant_server_round_trip() -> None:
    """Full round trip against the configured Qdrant server."""
    settings = get_settings()
    collection = f"it_{uuid.uuid4().hex[:8]}"
    store = QdrantVectorStore(
        url=settings.qdrant_url, collection=collection, dim=DIM
    )

    try:
        assert store.ensure_collection() is True
        chunks = _chunks()
        store.upsert_chunks(chunks, [_vector(0), _vector(1)])

        hits = store.search(query_vector=_vector(0), top_k=2)
        assert len(hits) == 2
        assert hits[0].chunk_id == "attendance_policy.pdf::p1::c0"
        assert hits[0].document_name == "attendance_policy.pdf"
        assert hits[0].page == 1
    finally:
        store.reset()
