"""Tests for the Qdrant vector store wrapper (against the in-memory fake)."""

from __future__ import annotations

import pytest

from app.ingestion import Chunk
from app.vectorstore import (
    PAYLOAD_KEYS,
    QdrantVectorStore,
    SearchResult,
    VectorStoreError,
    _point_id,
)
from tests.conftest import FakeQdrant, assert_payload_keys, make_payload_chunk


def make_chunk(index: int, text: str) -> Chunk:
    return Chunk(
        id=f"doc::p1::c{index}",
        document_name="doc.pdf",
        source_path="/data/doc.pdf",
        page=1,
        chunk_index=index,
        text=text,
    )


def test_point_id_is_deterministic() -> None:
    assert _point_id("doc::p1::c0") == _point_id("doc::p1::c0")
    assert _point_id("doc::p1::c0") != _point_id("doc::p1::c1")


def test_ensure_collection_creates_when_missing() -> None:
    client = FakeQdrant(dim=8)
    store = QdrantVectorStore(url="http://fake", collection="c", dim=8, client=client)
    assert store.ensure_collection() is True
    assert store.ensure_collection() is False  # second call verifies only


def test_ensure_collection_detects_dimension_mismatch() -> None:
    client = FakeQdrant(dim=8)
    store = QdrantVectorStore(url="http://fake", collection="c", dim=8, client=client)
    store.ensure_collection()

    other = QdrantVectorStore(url="http://fake", collection="c", dim=16, client=client)
    with pytest.raises(VectorStoreError, match="vector size 8"):
        other.ensure_collection()


def test_upsert_and_search_round_trip_with_metadata(fake_store) -> None:
    chunks = [make_chunk(0, "minimum attendance is seventy five percent"),
              make_chunk(1, "examination eligibility requires good standing")]
    vectors = [[0.1, 0.9] + [0.0] * 62, [0.9, 0.1] + [0.0] * 62]

    count = fake_store.store.upsert_chunks(chunks, vectors)
    assert count == 2

    hits = fake_store.store.search(query_vector=[0.1, 0.9] + [0.0] * 62, top_k=2)
    assert isinstance(hits[0], SearchResult)
    assert hits[0].chunk_id == "doc::p1::c0"
    assert hits[0].document_name == "doc.pdf"
    assert hits[0].source_path == "/data/doc.pdf"
    assert hits[0].page == 1
    assert hits[0].chunk_index == 0
    assert "attendance" in hits[0].text


def test_search_orders_by_similarity(fake_store) -> None:
    chunks = [make_chunk(0, "attendance"), make_chunk(1, "exams")]
    vectors = [[1.0] + [0.0] * 63, [0.0, 1.0] + [0.0] * 62]
    fake_store.store.upsert_chunks(chunks, vectors)

    hits = fake_store.store.search(query_vector=[0.0, 1.0] + [0.0] * 62, top_k=2)
    assert hits[0].chunk_id == "doc::p1::c1"
    assert hits[1].chunk_id == "doc::p1::c0"


def test_upsert_reingest_overwrites_same_chunk_ids(fake_store) -> None:
    chunk = make_chunk(0, "text")
    vector = [1.0] + [0.0] * 63
    fake_store.store.upsert_chunks([chunk], [vector])
    fake_store.store.upsert_chunks([chunk], [vector])
    assert len(fake_store.client.points) == 1


def test_upsert_rejects_chunk_vector_mismatch(fake_store) -> None:
    with pytest.raises(VectorStoreError, match="chunks but"):
        fake_store.store.upsert_chunks([make_chunk(0, "t")], [])


def test_upsert_empty_is_a_noop(fake_store) -> None:
    assert fake_store.store.upsert_chunks([], []) == 0


def test_search_rejects_non_positive_top_k(fake_store) -> None:
    with pytest.raises(VectorStoreError, match="top_k"):
        fake_store.store.search(query_vector=[0.0] * 64, top_k=0)


def test_search_surfaces_malformed_payloads(fake_store) -> None:
    fake_store.client.points["bad"] = {"vector": [0.0] * 64, "payload": {"text": "x"}}
    with pytest.raises(VectorStoreError, match="missing required keys"):
        fake_store.store.search(query_vector=[0.0] * 64, top_k=1)


def test_reset_deletes_collection(fake_store) -> None:
    fake_store.store.reset()
    assert fake_store.client.collection is None
    assert not fake_store.client.points


def test_payload_contains_all_metadata_keys() -> None:
    payload = make_payload_chunk("some text")
    assert_payload_keys(payload)
    for key in PAYLOAD_KEYS:
        assert payload[key] is not None


def test_invalid_dimension_is_rejected_at_construction() -> None:
    with pytest.raises(VectorStoreError, match="dimension"):
        QdrantVectorStore(url="http://fake", collection="c", dim=0, client=FakeQdrant())
