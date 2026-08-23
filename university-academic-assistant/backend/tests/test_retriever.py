"""Tests for the NormalRetriever (Phase 5)."""

import asyncio

from tests import rag_factory as factory
from tests.rag_factory import COLLECTIONS

from app.retrieval.normal_retriever import NormalRetriever


def _run(coro):
    return asyncio.run(coro)


def _indexed_retriever() -> NormalRetriever:
    service = factory.make_indexing_service()
    factory.index(
        service,
        factory.sample_structured(
            "Unit 1: Introduction\nDatabases store data.\n"
            "Unit 2: Normalization\nNormalization removes redundancy in tables.\n"
            "Unit 3: Transactions\nACID properties guarantee reliable processing."
        ),
    )
    return factory.make_retriever(service.embedder, service.repository, top_k=3)


def test_retrieve_returns_ranked_chunks_with_metadata() -> None:
    retriever = _indexed_retriever()
    results = _run(retriever.retrieve("normalization removes redundancy", top_k=3))

    assert results
    top = results[0]
    assert top.metadata["document_type"] == "syllabus"
    assert top.metadata["topic"] == "Normalization"
    assert "Normalization removes redundancy" in top.text
    assert "text" not in top.metadata  # text exposed separately
    assert top.collection == "university_docs"
    assert isinstance(top.score, float)
    assert all(results[i].score >= results[i + 1].score for i in range(len(results) - 1))


def test_retrieve_honours_top_k() -> None:
    retriever = _indexed_retriever()
    results = _run(retriever.retrieve("databases", top_k=1))
    assert len(results) == 1


def test_retrieve_supports_collection_filter() -> None:
    service = factory.make_indexing_service()
    factory.index(
        service,
        factory.sample_structured("Q1. Explain normalization. 10 Marks\nNormalization removes redundancy.", document_type="past_question"),
    )
    factory.index(
        service,
        factory.sample_structured("Unit 1: Introduction\nDatabases store data.", document_type="syllabus"),
    )
    retriever = factory.make_retriever(service.embedder, service.repository)

    results = _run(retriever.retrieve("normalization", collection="past_questions", top_k=3))
    assert results
    assert all(result.collection == "past_questions" for result in results)


def test_retrieve_empty_query_returns_nothing() -> None:
    retriever = _indexed_retriever()
    assert _run(retriever.retrieve("   ")) == []


def test_retrieve_min_score_gates_weak_matches() -> None:
    retriever = _indexed_retriever()
    # A zero-overlap query scores exactly 0 with the deterministic embedder and
    # is dropped by the relevance floor.
    assert _run(retriever.retrieve("What is the capital of France?")) == []


def test_retrieve_min_score_disabled_keeps_weak_matches() -> None:
    service = factory.make_indexing_service()
    factory.index(service, factory.sample_structured("Unit 1: Introduction\nDBMS is a database management system.\nDatabases store data."))
    retriever = factory.make_retriever(service.embedder, service.repository, min_score=0.0)
    results = _run(retriever.retrieve("What is the capital of France?"))
    assert results  # gating disabled -> nearest chunk still returned


def test_retrieve_empty_index_returns_nothing() -> None:
    service = factory.make_indexing_service()
    retriever = factory.make_retriever(service.embedder, service.repository)
    assert _run(retriever.retrieve("anything")) == []


def test_retrieve_missing_collection_is_skipped() -> None:
    retriever = NormalRetriever(
        embedder=factory.make_embedder(),
        repository=factory.make_repository(),
        collection_names=["does_not_exist"],
        top_k=3,
    )
    assert _run(retriever.retrieve("anything")) == []
    assert COLLECTIONS  # sanity: shared constant still defined