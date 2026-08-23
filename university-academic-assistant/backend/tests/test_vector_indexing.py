"""Tests for the vector indexing service (chunk -> embed -> index -> search)."""

import asyncio

from tests.structure_factory import make_structured

from app.chunking import SemanticChunker
from app.embedding import DeterministicEmbedder
from app.repositories.qdrant import QdrantRepository
from app.services.vector_indexing import VectorIndexingService
from app.structure.models import DocumentMetadata, DocumentStructure

COLLECTIONS = ["university_docs", "past_questions", "library_books"]


def _run(coro):
    return asyncio.run(coro)


def _service(**kwargs) -> VectorIndexingService:
    kwargs.setdefault(
        "embedder",
        DeterministicEmbedder(16),
    )
    kwargs.setdefault("repository", QdrantRepository(url=":memory:", collection_names=COLLECTIONS, default_dimensions=16))
    kwargs.setdefault("collection_names", COLLECTIONS)
    kwargs.setdefault(
        "chunker",
        SemanticChunker(max_chars=60, overlap_chars=0),
    )
    return VectorIndexingService(**kwargs)


def _syllabus():
    return make_structured(
        [
            "Unit 1: Introduction\nDatabases store data.\n"
            "Unit 2: Normalization\nNormalization removes redundancy in tables.\n"
            "Unit 3: Transactions\nACID properties guarantee reliable processing.",
        ],
        document_type="syllabus",
        metadata=DocumentMetadata(
            document_type="syllabus",
            title="DBMS Syllabus",
            semester=5,
            subject="Database Management System",
        ),
        structure=DocumentStructure(topics=["Introduction", "Normalization", "Transactions"]),
    )


def _past_paper():
    return make_structured(
        [
            "2080\nDatabase Management System\n"
            "Q1. Define DBMS. 5 Marks\nDBMS manages data.\n"
            "Q5. Explain normalization. 10 Marks\nNormalization removes redundancy.",
        ],
        document_type="past_question",
        metadata=DocumentMetadata(document_type="past_question", subject="Database Management System", year=2080),
    )


def test_index_syllabus_routes_to_university_docs() -> None:
    service = _service()
    result = _run(service.index(_syllabus()))

    assert result["chunk_count"] == 3
    assert result["collections"]["university_docs"] == 3
    assert _run(service.repository.count("university_docs")) == 3


def test_index_past_question_routes_to_past_questions() -> None:
    service = _service()
    result = _run(service.index(_past_paper()))

    assert result["collections"]["past_questions"] == result["chunk_count"]
    assert _run(service.repository.count("past_questions")) == result["chunk_count"]


def test_search_returns_relevant_normalization_chunk() -> None:
    service = _service()
    _run(service.index(_syllabus()))

    results = _run(service.search("normalization removes redundancy", top_k=3))

    assert results
    top = results[0]
    assert top.metadata["document_type"] == "syllabus"
    assert top.metadata["topic"] == "Normalization"
    assert "Normalization removes redundancy in tables." in top.text
    assert top.metadata["subject"] == "Database Management System"
    assert top.metadata["semester"] == 5
    assert top.collection == "university_docs"
    # Scores are sorted descending.
    assert all(results[i].score >= results[i + 1].score for i in range(len(results) - 1))


def test_search_with_collection_filter() -> None:
    service = _service()
    _run(service.index(_syllabus()))
    _run(service.index(_past_paper()))

    results = _run(service.search("define dbms", collection="past_questions", top_k=3))
    assert results
    assert all(result.collection == "past_questions" for result in results)


def test_search_empty_query_returns_nothing() -> None:
    service = _service()
    _run(service.index(_syllabus()))
    assert _run(service.search("   ")) == []


def test_health_reports_ok() -> None:
    service = _service()
    health = _run(service.health())
    assert health["status"] == "ok"
    assert health["embedder"] == "DeterministicEmbedder"
    assert health["collections"] == COLLECTIONS


def test_index_and_search_preserve_metadata() -> None:
    service = _service()
    _run(service.index(_syllabus()))

    results = _run(service.search("what is a database", top_k=1))
    assert results
    metadata = results[0].metadata
    assert metadata["document_id"] == "doc-1"
    assert metadata["page"] == 1
    assert "text" not in metadata  # text is exposed separately