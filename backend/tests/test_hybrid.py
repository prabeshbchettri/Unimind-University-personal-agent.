"""Tests for the hybrid (vector + BM25) retrieval strategy.

Two layers:

1. fusion logic is exercised precisely with stubs for the two sources so
   ordering, deduplication and metadata are asserted exactly;
2. one test stitches the real ``VectorRetriever`` + real ``BM25Index`` + real
   RRF fusion together to prove a lexical-only chunk genuinely surfaces (only
   the embedding and the Qdrant engine are the project's deterministic fakes).
"""

from __future__ import annotations

import pytest

from app.bm25 import BM25Index, build_bm25_index
from app.ingestion import Chunk
from app.retriever import HybridRetriever, RetrievalError, VectorRetriever
from app.vectorstore import QdrantVectorStore
from tests.conftest import FakeEmbedding, FakeQdrant, make_retrieved


def make_chunk(chunk_id, text, document="doc.pdf", page=1, index=0) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_name=document,
        source_path=f"data/documents/{document}",
        page=page,
        chunk_index=index,
        text=text,
    )


def build_bm25(chunks: list[Chunk]) -> BM25Index:
    index = BM25Index()
    for chunk in chunks:
        index.add(chunk)
    return index


class StubVector:
    """Stand-in for VectorRetriever that returns a fixed ranked list."""

    def __init__(self, results) -> None:
        self.results = results
        self.queries: list[str] = []

    def retrieve(self, query: str) -> list:
        self.queries.append(query)
        return self.results


@pytest.fixture
def lexical_index() -> BM25Index:
    return build_bm25(
        [
            make_chunk(
                "c0",
                "cs201 syllabus covers trees and graphs",
                document="syllabus.pdf",
                page=2,
                index=1,
            )
        ]
    )


def test_hybrid_combines_vector_and_lexical_candidates(lexical_index) -> None:
    vector = StubVector(
        [
            make_retrieved("attendance 75 percent", document="a.pdf", score=0.8, chunk_id="a0"),
            make_retrieved("library six books", document="b.pdf", score=0.7, chunk_id="b0"),
        ]
    )
    hybrid = HybridRetriever(vector=vector, bm25=lexical_index, top_k=3)

    results = hybrid.retrieve("cs201 syllabus trees")

    ids = [chunk.chunk_id for chunk in results]
    assert "a0" in ids and "b0" in ids and "c0" in ids
    assert vector.queries == ["cs201 syllabus trees"]


def test_hybrid_preserves_source_metadata_for_lexical_only_chunk(lexical_index) -> None:
    vector = StubVector(
        [make_retrieved("attendance 75 percent", document="a.pdf", score=0.8, chunk_id="a0")]
    )
    hybrid = HybridRetriever(vector=vector, bm25=lexical_index, top_k=3)

    lexical_only = next(
        chunk for chunk in hybrid.retrieve("cs201 trees") if chunk.chunk_id == "c0"
    )

    # Metadata came from the BM25 index, not the vector store.
    assert lexical_only.document == "syllabus.pdf"
    assert lexical_only.page == 2
    assert lexical_only.chunk_index == 1
    assert lexical_only.semantic_score is None
    assert lexical_only.bm25_score is not None
    assert lexical_only.score == lexical_only.bm25_score


def test_hybrid_deduplicates_chunks_retrieved_by_both_strategies() -> None:
    vector = StubVector(
        [
            make_retrieved("shared passage", chunk_id="x0", score=0.9),
            make_retrieved("another passage", chunk_id="y0", score=0.6),
        ]
    )
    bm25 = build_bm25([make_chunk("x0", "shared passage"), make_chunk("z0", "unique match")])
    hybrid = HybridRetriever(vector=vector, bm25=bm25, top_k=5)

    results = hybrid.retrieve("shared unique")

    ids = [chunk.chunk_id for chunk in results]
    assert len(ids) == len(set(ids)), "no chunk may appear twice after fusion"
    assert ids.count("x0") == 1
    shared = next(chunk for chunk in results if chunk.chunk_id == "x0")
    assert shared.semantic_score is not None
    assert shared.bm25_score is not None


def test_hybrid_ranks_by_rrf_not_raw_score_concatenation() -> None:
    # x1 is the top vector hit; y1 is second in vector BUT first in BM25, so
    # RRF must put y1 above x1 (not simply vector order followed by BM25).
    vector = StubVector(
        [
            make_retrieved("alpha beta", chunk_id="x1", score=0.99),
            make_retrieved("beta", chunk_id="y1", score=0.5),
        ]
    )
    bm25 = build_bm25([make_chunk("y1", "beta target term"), make_chunk("w1", "other words")])
    hybrid = HybridRetriever(vector=vector, bm25=bm25, top_k=5)

    ids = [chunk.chunk_id for chunk in hybrid.retrieve("beta target")]
    assert ids[0] == "y1"
    assert "x1" in ids


def test_hybrid_honors_top_k(lexical_index) -> None:
    vector = StubVector(
        [
            make_retrieved("one", chunk_id="a0", score=0.9),
            make_retrieved("two", chunk_id="b0", score=0.8),
            make_retrieved("three", chunk_id="d0", score=0.7),
        ]
    )
    hybrid = HybridRetriever(vector=vector, bm25=lexical_index, top_k=2)

    assert len(hybrid.retrieve("cs201 trees")) == 2


def test_hybrid_rejects_empty_query(lexical_index) -> None:
    hybrid = HybridRetriever(vector=StubVector([]), bm25=lexical_index, top_k=3)
    with pytest.raises(RetrievalError, match="empty"):
        hybrid.retrieve("   ")


def make_chunk_named(name: str, text: str) -> Chunk:
    return Chunk(
        id=f"{name}::p1::c0",
        document_name=name,
        source_path=f"data/documents/{name}",
        page=1,
        chunk_index=0,
        text=text,
    )


def test_real_vector_plus_bm25_surfaces_lexical_only_chunk() -> None:
    """Real stack (except embedding/Qdrant fakes): a chunk only BM25 finds
    must still reach the fused results with its metadata."""
    embedding = FakeEmbedding(dim=64)
    client = FakeQdrant(dim=64)
    store = QdrantVectorStore(url="http://fake", collection="test_docs", dim=64, client=client)
    store.ensure_collection()

    docs = {
        "pdf_a.pdf": "xylophone concert hall",
        "pdf_b.pdf": "quarterly report schedule",
        "pdf_c.pdf": "zanzibar island excursion",
    }
    chunks = [make_chunk_named(name, text) for name, text in docs.items()]
    store.upsert_chunks(chunks, [embedding.embed_query(c.text) for c in chunks])

    vector = VectorRetriever(store=store, embedding=embedding, top_k=2)
    bm25 = build_bm25_index(store)
    hybrid = HybridRetriever(vector=vector, bm25=bm25, top_k=3)

    results = hybrid.retrieve("xylophone quarterly zanzibar")

    assert len(results) == 3  # both strategies' candidates, deduplicated
    lexical_only = [chunk for chunk in results if chunk.semantic_score is None]
    assert len(lexical_only) == 1  # the doc outside the vector top-2 is BM25-only
    hit = lexical_only[0]
    assert hit.document in docs
    assert hit.bm25_score is not None
    assert hit.text