"""Tests for the Qdrant repository and collection routing."""

import asyncio

from app.repositories.qdrant import QdrantRepository, point_uuid
from app.retrieval import all_collections, collection_for


def _run(coro):
    return asyncio.run(coro)


def _repo(**kwargs) -> QdrantRepository:
    kwargs.setdefault("url", ":memory:")
    kwargs.setdefault("collection_names", ["university_docs", "past_questions", "library_books"])
    kwargs.setdefault("default_dimensions", 4)
    return QdrantRepository(**kwargs)


def test_ping_in_memory() -> None:
    repo = _repo()
    assert _run(repo.ping()) is True
    _run(repo.close())


def test_ensure_collections_creates_three() -> None:
    repo = _repo()
    _run(repo.ensure_collections())
    assert _run(repo.collection_exists("university_docs"))
    assert _run(repo.collection_exists("past_questions"))
    assert _run(repo.collection_exists("library_books"))
    _run(repo.close())


def test_upsert_and_count() -> None:
    repo = _repo()
    _run(repo.ensure_collections())
    _run(
        repo.upsert_points(
            "university_docs",
            [
                {"id": point_uuid("a"), "vector": [1.0, 0, 0, 0], "payload": {"text": "hello", "document_id": "d1"}},
                {"id": point_uuid("b"), "vector": [0.0, 1, 0, 0], "payload": {"text": "bye", "document_id": "d2"}},
            ],
        )
    )
    assert _run(repo.count("university_docs")) == 2
    _run(repo.close())


def test_search_returns_ordered_hits() -> None:
    repo = _repo()
    _run(repo.ensure_collections())
    _run(
        repo.upsert_points(
            "university_docs",
            [
                {"id": point_uuid("a"), "vector": [1.0, 0, 0, 0], "payload": {"text": "hello"}},
                {"id": point_uuid("b"), "vector": [0.0, 1, 0, 0], "payload": {"text": "bye"}},
            ],
        )
    )
    hits = _run(repo.search("university_docs", [1.0, 0, 0, 0], top_k=1))
    assert len(hits) == 1
    assert hits[0].payload["text"] == "hello"
    assert hits[0].score == 1.0
    _run(repo.close())


def test_upsert_is_idempotent() -> None:
    repo = _repo()
    _run(repo.ensure_collections())
    point = {"id": point_uuid("a"), "vector": [1.0, 0, 0, 0], "payload": {"text": "x"}}
    _run(repo.upsert_points("university_docs", [point]))
    _run(repo.upsert_points("university_docs", [point]))
    assert _run(repo.count("university_docs")) == 1
    _run(repo.close())


def test_collection_for_routing() -> None:
    names = ["university_docs", "past_questions", "library_books"]
    assert collection_for("syllabus", names) == "university_docs"
    assert collection_for("notice", names) == "university_docs"
    assert collection_for("rules_regulations", names) == "university_docs"
    assert collection_for("academic_calendar", names) == "university_docs"
    assert collection_for("unknown", names) == "university_docs"
    assert collection_for("past_question", names) == "past_questions"
    assert collection_for("library_book", names) == "library_books"


def test_all_collections_order() -> None:
    assert all_collections(["a", "b", "c"]) == ["a", "b", "c"]
    assert all_collections(()) == ["university_docs", "past_questions", "library_books"]


def test_point_uuid_is_stable() -> None:
    assert str(point_uuid("chunk-1")) == str(point_uuid("chunk-1"))
    assert point_uuid("chunk-1") != point_uuid("chunk-2")