"""Tests for the deterministic BM25 index (pure logic, no services)."""

from __future__ import annotations

import pytest

from app.bm25 import BM25Index, build_bm25_index, tokenize
from app.ingestion import Chunk


def make_chunk(chunk_id: str, text: str, document: str = "doc.pdf", page: int = 1) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_name=document,
        source_path=f"data/documents/{document}",
        page=page,
        chunk_index=0,
        text=text,
    )


def test_tokenize_lowercases_and_keeps_alphanumerics() -> None:
    assert tokenize("ENCT 353: prerequisites for CS201.") == [
        "enct", "353", "prerequisites", "for", "cs201",
    ]


def test_index_ranks_exact_term_match_first() -> None:
    index = BM25Index()
    index.add(make_chunk("c1", "library loans last fourteen days"))
    index.add(make_chunk("c2", "policy and regulations apply to attendance"))
    index.add(make_chunk("c3", "cs201 credit requirement"))

    hits = index.search("cs201 credit policy")
    assert hits[0].chunk_id == "c3"  # two exact query terms
    assert hits[0].document == "doc.pdf"
    assert hits[0].text.startswith("cs201")
    assert hits[1].chunk_id == "c2"  # one exact query term
    assert hits[0].score > hits[1].score > 0


def test_search_is_deterministic() -> None:
    index = BM25Index()
    for i, text in enumerate(["a b c d", "b c d e", "c c d e f"]):
        index.add(make_chunk(f"c{i}", text))

    first = [h.chunk_id for h in index.search("c d", top_k=3)]
    second = [h.chunk_id for h in index.search("c d", top_k=3)]
    assert first == second
    assert first[0] == "c2"  # highest term frequency for both query terms


def test_search_ignores_unknown_terms_and_respects_top_k() -> None:
    index = BM25Index()
    index.add(make_chunk("c1", "attendance rules"))
    index.add(make_chunk("c2", "library fines"))

    assert index.search("zzz nonexistent") == []  # no term matched -> no hits
    hits = index.search("attendance library fines", top_k=1)
    assert len(hits) == 1


def test_search_on_empty_index_returns_empty() -> None:
    assert BM25Index().search("anything") == []


def test_search_rejects_non_positive_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        BM25Index().search("query", top_k=0)


def test_add_twice_with_same_id_replaces() -> None:
    index = BM25Index()
    index.add(make_chunk("c1", "old text"))
    index.add(make_chunk("c1", "new phrase"))
    assert index.size == 1
    hits = index.search("phrase")
    assert hits[0].chunk_id == "c1"


def test_build_bm25_index_from_store_reads_payloads(fake_store) -> None:
    """End-to-end: BM25 index built from real vector store payloads."""
    store = fake_store.store

    chunk = Chunk(
        id="lp::p1::c0",
        document_name="library.pdf",
        source_path="data/documents/library.pdf",
        page=1,
        chunk_index=0,
        text="loan period is fourteen days",
    )
    store.upsert_chunks([chunk], [[1.0] + [0.0] * 63])

    index = build_bm25_index(store)
    assert index.size == 1
    hits = index.search("loan period")
    assert hits[0].chunk_id == "lp::p1::c0"
    assert hits[0].document == "library.pdf"
    assert hits[0].page == 1
    assert hits[0].chunk_index == 0