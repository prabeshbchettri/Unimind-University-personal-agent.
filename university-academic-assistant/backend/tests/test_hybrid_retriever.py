"""Tests for Phase 6 hybrid retrieval.

Covers the BM25 sparse index, the HybridRetriever (dense + sparse, RRF fusion,
rerank), metadata preservation, and a dense-vs-hybrid comparison across the
seven representative query types: semantic, subject code, regulation number,
date, academic year, past-question number, and exact phrase.
"""

import asyncio

import pytest

from tests import rag_factory as factory
from tests.rag_factory import COLLECTIONS

from app.retrieval import SearchResult
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.normal_retriever import NormalRetriever
from app.retrieval.sparse import BM25Index, tokenize
from app.services.chat import ChatService


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Corpus: one document per representative exact-match case (unique ids).
# ---------------------------------------------------------------------------

def _corpus():
    return [
        factory.sample_structured(
            "Unit 1: Introduction\nDatabases store data.\n"
            "Unit 2: Normalization\nNormalization removes redundancy in tables.",
            document_type="syllabus",
            title="DBMS Syllabus",
            topics=["Introduction", "Normalization"],
            document_id="doc-syllabus",
        ),
        factory.sample_structured(
            "Rules and Regulations\nRegulation 12\nStudents must attend at least 75 percent of classes to sit the final exam.",
            document_type="rules_regulations",
            title="Rules and Regulations",
            topics=["Regulation 12"],
            document_id="doc-reg",
        ),
        factory.sample_structured(
            "Course Code CSIT 325\nThis course introduces advanced databases.",
            document_type="syllabus",
            title="CSIT 325 Syllabus",
            topics=["CSIT 325"],
            document_id="doc-csit",
        ),
        factory.sample_structured(
            "Notice dated 2024-05-01\nClasses are cancelled on 2024-05-01.",
            document_type="notice",
            title="May Notice",
            topics=["Notice"],
            document_id="doc-notice",
        ),
        factory.sample_structured(
            "2080\nQ5. Explain normalization with an example. 10 Marks\nNormalization is the process of organizing data to minimize redundancy.",
            document_type="past_question",
            title="DBMS 2080",
            document_id="doc-past",
        ),
    ]


def _build(min_score: float = factory.DEFAULT_MIN_SCORE):
    """Return (dense, hybrid, sparse_index) against the sample corpus."""
    service = factory.make_indexing_service(sparse_index=factory.make_sparse_index())
    for structured in _corpus():
        factory.index(service, structured)
    dense = factory.make_retriever(service.embedder, service.repository, top_k=3, min_score=min_score)
    hybrid = factory.make_hybrid_retriever(dense, service.sparse_index, top_k=3)
    return dense, hybrid, service.sparse_index


EXPECTED_TITLES = {
    "Explain normalization.": "DBMS Syllabus",
    "What is CSIT 325 about?": "CSIT 325 Syllabus",
    "What does regulation 12 say about attendance?": "Rules and Regulations",
    "What happened on 2024-05-01?": "May Notice",
    "Which year past paper is 2080?": "DBMS 2080",
    "What is question 5 about?": "DBMS 2080",
    "normal forms 1NF 2NF 3NF": "DBMS 2080",
}


# ---------------------------------------------------------------------------
# BM25 sparse index
# ---------------------------------------------------------------------------

def test_tokenize_keeps_codes_and_numbers() -> None:
    tokens = tokenize("Regulation 12 says CSIT 325 on 2024-05-01")
    assert "regulation" in tokens
    assert "12" in tokens
    assert "csit" in tokens
    assert "325" in tokens
    assert "2024" in tokens and "05" in tokens and "01" in tokens
    assert "the" not in tokens


def test_bm25_finds_exact_term_match() -> None:
    index = factory.make_sparse_index()
    index.add_chunks(
        [
            factory.make_chunk("Regulation 12: Students must attend 75 percent of classes.", document_id="doc-a", topic="attendance"),
            factory.make_chunk("Normalization removes redundancy.", document_id="doc-b", topic="normalization"),
        ],
        "university_docs",
    )
    hits = index.search("regulation 12 attendance", top_k=5)
    assert hits
    assert hits[0].chunk_id.startswith("doc-a")
    assert hits[0].score > 0
    assert hits[0].collection == "university_docs"


def test_bm25_no_overlap_returns_nothing() -> None:
    index = factory.make_sparse_index()
    index.add_chunks(
        [factory.make_chunk("Normalization removes redundancy.", document_id="doc-b", topic="normalization")],
        "university_docs",
    )
    assert index.search("zyzzx nope", top_k=5) == []
    assert index.search("   ", top_k=5) == []


def test_bm25_preserves_payload_metadata() -> None:
    index = factory.make_sparse_index()
    index.add(
        factory.make_chunk(
            "Course Code CSIT 325.",
            document_id="doc-csit",
            title="CSIT 325 Syllabus",
            subject="Advanced Database",
            topic="CSIT 325",
        ),
        "university_docs",
    )
    hits = index.search("CSIT 325", top_k=5)
    assert hits
    payload = hits[0].payload
    assert payload["document_id"] == "doc-csit"
    assert payload["document_type"] == "syllabus"
    assert payload["title"] == "CSIT 325 Syllabus"
    assert payload["page"] == 1
    assert payload["subject"] == "Advanced Database"
    assert payload["topic"] == "CSIT 325"


def test_bm25_invalid_params_rejected() -> None:
    with pytest.raises(ValueError):
        BM25Index(k1=0)
    with pytest.raises(ValueError):
        BM25Index(b=2)


# ---------------------------------------------------------------------------
# HybridRetriever building blocks
# ---------------------------------------------------------------------------

def test_search_dense_marks_method() -> None:
    dense, hybrid, _ = _build()
    results = _run(hybrid.search_dense("Explain normalization.", top_k=3))
    assert results
    assert all(result.method == "dense" for result in results)
    assert results[0].metadata["title"] == "DBMS Syllabus"


def test_search_sparse_marks_method_and_preserves_metadata() -> None:
    _, hybrid, _ = _build()
    results = hybrid.search_sparse("regulation 12", top_k=5)
    assert results
    assert all(result.method == "sparse" for result in results)
    top = results[0]
    assert top.metadata["document_id"] == "doc-reg"
    assert top.metadata["title"] == "Rules and Regulations"
    assert top.metadata["document_type"] == "rules_regulations"
    assert top.collection == "university_docs"
    assert "Regulation 12" in top.text


def test_fuse_rrf_boosts_chunks_found_by_both_systems() -> None:
    shared = factory.normalization_chunk()
    shared.metadata["chunk_id"] = "chunk-1"
    only_dense = factory.SearchResult(
        text="Databases store data.",
        score=0.9,
        metadata={"chunk_id": "chunk-2", "title": "DBMS Syllabus"},
        collection="university_docs",
        method="dense",
    )
    only_sparse = factory.SearchResult(
        text="Course Code CSIT 325.",
        score=50.0,
        metadata={"chunk_id": "chunk-3", "title": "CSIT 325 Syllabus"},
        collection="university_docs",
        method="sparse",
    )
    sparse_copy = factory.SearchResult(
        text=shared.text,
        score=80.0,
        metadata=dict(shared.metadata),
        collection=shared.collection,
        method="sparse",
    )

    dense_list = [shared, only_dense]
    sparse_list = [sparse_copy, only_sparse]
    hybrid = _hybrid()
    fused = hybrid.fuse_results(dense_list, sparse_list)

    by_key = {entry.key: entry for entry in fused}
    shared_entry = by_key[("university_docs", "chunk-1")]
    only_dense_entry = by_key[("university_docs", "chunk-2")]
    # Appearing in both lists earns a larger RRF score than appearing in one.
    assert shared_entry.rrf_score > only_dense_entry.rrf_score
    assert shared_entry.dense_score == 0.9
    assert shared_entry.sparse_score == 80.0


def test_rerank_sorts_and_tags_hybrid() -> None:
    hybrid = _hybrid()
    fused = hybrid.fuse_results(
        [factory.SearchResult(text="a", score=0.8, metadata={"chunk_id": "1", "title": "A"}, collection="university_docs", method="dense")],
        [factory.SearchResult(text="b", score=5.0, metadata={"chunk_id": "2", "title": "B"}, collection="university_docs", method="sparse")],
    )
    ranked = hybrid.rerank_results(fused, top_k=5)
    assert len(ranked) == 2
    assert all(result.method == "hybrid" for result in ranked)
    assert ranked[0].score >= ranked[1].score


def test_retrieve_returns_hybrid_results() -> None:
    _, hybrid, _ = _build()
    results = _run(hybrid.retrieve("Explain normalization.", top_k=3))
    assert results
    assert len(results) <= 3
    assert all(result.method == "hybrid" for result in results)
    assert results[0].metadata["title"] == "DBMS Syllabus"


def test_retrieve_empty_query_returns_nothing() -> None:
    _, hybrid, _ = _build()
    assert _run(hybrid.retrieve("   ")) == []


def _hybrid() -> HybridRetriever:
    index = factory.make_sparse_index()
    dense = factory.make_retriever(factory.make_embedder(), factory.make_repository(), top_k=3)
    return factory.make_hybrid_retriever(dense, index, top_k=3)


# ---------------------------------------------------------------------------
# Dense vs hybrid comparison (acceptance)
# ---------------------------------------------------------------------------

def test_semantic_query_works_in_both() -> None:
    dense, hybrid, _ = _build()
    dense_top = _run(dense.retrieve("Explain normalization.", top_k=3))
    hybrid_top = _run(hybrid.retrieve("Explain normalization.", top_k=3))
    assert dense_top[0].metadata["title"] == "DBMS Syllabus"
    assert hybrid_top[0].metadata["title"] == "DBMS Syllabus"


@pytest.mark.parametrize(
    "query",
    [
        "What is CSIT 325 about?",
        "What does regulation 12 say about attendance?",
        "What happened on 2024-05-01?",
        "Which year past paper is 2080?",
        "What is question 5 about?",
        "normal forms 1NF 2NF 3NF",
    ],
)
def test_exact_match_queries_hit_expected_source_in_hybrid(query: str) -> None:
    dense, hybrid, _ = _build()
    expected = EXPECTED_TITLES[query]
    hybrid_top = _run(hybrid.retrieve(query, top_k=3))
    titles = [result.metadata.get("title") for result in hybrid_top]
    assert expected in titles, f"{query!r}: hybrid top-3 = {titles}"


def test_hybrid_improves_regulation_exact_match() -> None:
    dense, hybrid, _ = _build()
    query = "What does regulation 12 say about attendance?"
    dense_top = _run(dense.retrieve(query, top_k=3))
    hybrid_top = _run(hybrid.retrieve(query, top_k=3))
    # Dense finds the regulation chunk but weakly; hybrid promotes it to #1.
    assert hybrid_top[0].metadata["title"] == "Rules and Regulations"
    assert hybrid_top[0].score > dense_top[0].score


def test_hybrid_recovers_when_dense_has_no_confident_match() -> None:
    # A strict relevance floor gates the weak dense match, but BM25 still
    # finds the exact regulation chunk, so hybrid answers it.
    dense, hybrid, _ = _build(min_score=0.5)
    query = "What does regulation 12 say about attendance?"
    assert _run(dense.retrieve(query, top_k=3)) == []
    hybrid_top = _run(hybrid.retrieve(query, top_k=3))
    assert hybrid_top
    assert hybrid_top[0].metadata["title"] == "Rules and Regulations"


def test_normal_retriever_still_works() -> None:
    dense, _, _ = _build()
    results = _run(dense.retrieve("Explain normalization.", top_k=3))
    assert results
    assert results[0].metadata["title"] == "DBMS Syllabus"
    assert results[0].method == "dense"


# ---------------------------------------------------------------------------
# Chat-level integration with the hybrid retriever
# ---------------------------------------------------------------------------

def test_chat_answers_from_hybrid_sources() -> None:
    service = factory.make_indexing_service(sparse_index=factory.make_sparse_index())
    for structured in _corpus():
        factory.index(service, structured)
    dense = factory.make_retriever(service.embedder, service.repository, top_k=3)
    hybrid = factory.make_hybrid_retriever(dense, service.sparse_index, top_k=3)
    chat = ChatService(
        retriever=hybrid,
        context_builder=factory.make_context_builder(),
        llm=factory.make_llm(),
        top_k=3,
    )
    result = _run(chat.answer("What does regulation 12 say about attendance?"))
    assert result.sources
    assert result.sources[0].metadata["title"] == "Rules and Regulations"
    assert result.sources[0].method == "hybrid"