"""Tests for the syllabus-aware library book recommendation (Phase 9)."""

import asyncio
import json

from app.recommendation.coverage import CoverageCalculator, normalize_name, rank
from app.recommendation.models import BookProfile
from app.recommendation.resolver import TopicResolver
from app.llm.stub import StubLLMClient
from tests import rag_factory
from tests.rag_factory import (
    make_graph_service,
    make_library_book,
    make_recommendation_service,
    make_sparse_index,
)


def _run(coro):
    return asyncio.run(coro)


SYLLABUS_TEXT = (
    "Database Management System Syllabus\n"
    "Unit 1: Introduction\nDatabases store data.\n"
    "Unit 2: Normalization\nNormalization removes redundancy in tables.\n"
    "Unit 3: Transactions\nACID properties keep transactions safe.\n"
    "Unit 4: Indexing\nIndexes speed up queries.\n"
)

BOOK_STRONG = ("Database System Concepts", [
    ("Introduction", ["Data Models"]),
    ("Normalization", ["Normal Forms", "First Normal Form"]),
    ("Transactions", ["ACID Properties"]),
    ("Indexing", ["B-Trees"]),
])
BOOK_PARTIAL = ("Database Fundamentals", [
    ("Normalization", ["Normal Forms"]),
    ("Introduction", []),
])
BOOK_POOR = ("Operating System Principles", [
    ("Processes", ["Scheduling"]),
    ("Memory", ["Paging"]),
])


# ---------------------------------------------------------------------------
# Coverage calculator (transparent scoring)
# ---------------------------------------------------------------------------

def test_normalize_name() -> None:
    assert normalize_name("  Normalization! ") == "normalization"
    assert normalize_name("Database  Systems") == "database systems"


def test_coverage_score_formula_is_transparent() -> None:
    calculator = CoverageCalculator(embedder=None, topic_weight=0.85, subtopic_weight=0.15)
    book = BookProfile(
        title="Book",
        document_id="b",
        topics=["Normalization", "Transactions", "Indexing"],
        subtopics=["Normal Forms", "ACID"],
    )
    coverage = calculator.calculate(
        book,
        required_topics=["Normalization", "Transactions", "Indexing", "Distributed Databases"],
        required_subtopics=["Normal Forms", "ACID"],
    )
    # topic_coverage = 3/4 = 0.75; subtopic_coverage = 2/2 = 1.0
    # score = 0.85 * 0.75 + 0.15 * 1.0 = 0.7875
    assert coverage.topic_coverage == 0.75
    assert coverage.subtopic_coverage == 1.0
    assert coverage.score == 0.7875
    assert coverage.matched_topics == ["Normalization", "Transactions", "Indexing"]
    assert coverage.missing_topics == ["Distributed Databases"]
    assert coverage.matched_subtopics == ["Normal Forms", "ACID"]
    assert coverage.evidence == ["Book", "b"]
    assert coverage.chapter_count == 3


def test_coverage_without_subtopics_ignores_subtopic_component() -> None:
    calculator = CoverageCalculator(embedder=None)
    coverage = calculator.calculate(
        BookProfile(title="Book", document_id="b", topics=["Normalization"]),
        required_topics=["Normalization"],
        required_subtopics=[],
    )
    # subtopic_coverage = 0 when there are no required subtopics: score = 1.0 * 0.85
    assert coverage.score == 0.85
    assert coverage.subtopic_coverage == 0.0


def test_semantic_match_uses_embedding_similarity() -> None:
    embedder = rag_factory.make_embedder()
    calculator = CoverageCalculator(embedder=embedder, semantic_threshold=0.4)
    coverage = calculator.calculate(
        BookProfile(title="Book", document_id="b", topics=["Normalization"]),
        required_topics=["Database Normalization"],
        required_subtopics=[],
    )
    assert coverage.matched_topics == ["Database Normalization"]


def test_exact_only_without_embedder() -> None:
    calculator = CoverageCalculator(embedder=None)
    coverage = calculator.calculate(
        BookProfile(title="Book", document_id="b", topics=["Normalization"]),
        required_topics=["Database Normalization"],
        required_subtopics=[],
    )
    assert coverage.matched_topics == []
    assert coverage.score == 0.0


def test_empty_required_topics_scores_zero() -> None:
    calculator = CoverageCalculator(embedder=None)
    coverage = calculator.calculate(BookProfile(title="Book", document_id="b"), [], [])
    assert coverage.score == 0.0
    assert coverage.missing_topics == []


def test_rank_orders_by_score_then_matches_then_title() -> None:
    a = BookProfile(title="Alpha", document_id="a", topics=["Normalization"])
    b = BookProfile(title="Beta", document_id="b", topics=["Normalization", "Transactions"])
    calculator = CoverageCalculator(embedder=None)
    coverages = [
        calculator.calculate(profile, ["Normalization", "Transactions"], [])
        for profile in (a, b)
    ]
    ordered = rank(coverages)
    assert [c.book_title for c in ordered] == ["Beta", "Alpha"]
    assert ordered[0].score > ordered[1].score


# ---------------------------------------------------------------------------
# Topic resolver (syllabus evidence)
# ---------------------------------------------------------------------------

def _indexed_environment():
    service = rag_factory.make_indexing_service(
        graph_service=make_graph_service(),
        sparse_index=make_sparse_index(),
    )
    rag_factory.index(
        service,
        rag_factory.sample_structured(SYLLABUS_TEXT, document_id="doc-syllabus"),
    )
    for document_id, (title, chapters) in [
        ("doc-book-strong", BOOK_STRONG),
        ("doc-book-partial", BOOK_PARTIAL),
        ("doc-book-poor", BOOK_POOR),
    ]:
        rag_factory.index(
            service,
            make_library_book(document_id=document_id, title=title, chapters=chapters),
        )
    return service


def test_resolver_identifies_required_topics_from_syllabus() -> None:
    service = _indexed_environment()
    resolver = TopicResolver(embedder=service.embedder, repository=service.repository)
    context = _run(resolver.identify("Which book is best for normalization?"))
    assert context.no_syllabus_evidence is False
    assert "Normalization" in context.required_topics
    assert context.required_topics
    assert context.reason


def test_resolver_no_syllabus_evidence() -> None:
    service = _indexed_environment()
    resolver = TopicResolver(embedder=service.embedder, repository=service.repository)
    context = _run(resolver.identify("quantum physics research"))
    assert context.no_syllabus_evidence is True
    assert context.required_topics == []


# ---------------------------------------------------------------------------
# Recommendation service end-to-end
# ---------------------------------------------------------------------------

def test_recommendation_ranks_strong_partial_poor_books() -> None:
    service = _indexed_environment()
    recommender = make_recommendation_service(
        repository=service.repository,
        embedder=service.embedder,
        graph_service=service.graph_service,
    )
    result = _run(recommender.recommend("Which book is best for normalization?"))
    assert result.no_syllabus_evidence is False
    assert result.subject == "Database Management System"
    assert "Normalization" in result.required_topics

    ordered = result.recommendations
    assert [c.book_title for c in ordered] == [
        "Database System Concepts",
        "Database Fundamentals",
        "Operating System Principles",
    ]
    strong, partial, poor = ordered
    assert strong.score > partial.score > poor.score
    assert "Normalization" in strong.matched_topics
    assert "Transactions" in strong.matched_topics
    assert "Distributed Databases" not in strong.matched_topics  # not required
    assert "Normalization" in partial.matched_topics
    assert poor.matched_topics == []
    assert all(c.evidence for c in ordered)
    assert "covers" in result.explanation


def test_recommendation_absent_topic_reports_no_evidence() -> None:
    service = _indexed_environment()
    recommender = make_recommendation_service(repository=service.repository, embedder=service.embedder)
    result = _run(recommender.recommend("recommend a book about quantum physics"))
    assert result.no_syllabus_evidence is True
    assert result.recommendations == []
    assert "No syllabus evidence" in result.explanation


def test_recommendation_llm_explanation_is_grounded() -> None:
    service = _indexed_environment()
    seen = {}

    def responder(prompt: str) -> str:
        seen["prompt"] = prompt
        return "The top book covers the required topics."

    llm = StubLLMClient(responder=responder)
    recommender = make_recommendation_service(
        repository=service.repository,
        embedder=service.embedder,
        llm=llm,
    )
    result = _run(recommender.recommend("Which book is best for normalization?"))
    assert result.explanation == "The top book covers the required topics."
    evidence = json.loads(seen["prompt"])
    assert "recommendations" in evidence
    assert evidence["recommendations"][0]["book"] == "Database System Concepts"


def test_recommendation_health_reports_formula() -> None:
    recommender = make_recommendation_service(repository=rag_factory.make_repository(), embedder=rag_factory.make_embedder())
    health = _run(recommender.health())
    assert health["status"] == "ok"
    assert "coverage = 0.85 * topic + 0.15 * subtopic" in health["calculator"]


def test_list_points_returns_library_book_payloads() -> None:
    service = _indexed_environment()
    payloads = _run(service.repository.list_points("library_books"))
    assert payloads
    for payload in payloads:
        assert payload["document_type"] == "library_book"
    topics = {payload.get("topic") for payload in payloads}
    assert "Normalization" in topics


# ---------------------------------------------------------------------------
# API integration
# ---------------------------------------------------------------------------

def _index_pdf(client, make_text_pdf, tmp_path, filename: str, text: str) -> None:
    path = make_text_pdf(tmp_path / filename, [text])
    with open(path, "rb") as handle:
        response = client.post(
            "/documents/index",
            files={"file": (filename, handle, "application/pdf")},
        )
    assert response.status_code == 200


def test_recommendations_endpoint_returns_ranked_books(client, make_text_pdf, tmp_path) -> None:
    _index_pdf(client, make_text_pdf, tmp_path, "syllabus.pdf", SYLLABUS_TEXT)
    _index_pdf(
        client, make_text_pdf, tmp_path, "book_strong.pdf",
        "Chapter 1: Introduction\n1.1 Data Models\n"
        "Chapter 2: Normalization\n2.1 Normal Forms\n"
        "Chapter 3: Transactions\n3.1 ACID Properties\n"
        "Chapter 4: Indexing\n4.1 B-Trees\n",
    )
    _index_pdf(
        client, make_text_pdf, tmp_path, "book_partial.pdf",
        "Chapter 1: Normalization\n1.1 Normal Forms\n",
    )

    response = client.post("/recommendations", json={"query": "Which book is best for normalization?"})
    assert response.status_code == 200
    body = response.json()
    assert body["no_syllabus_evidence"] is False
    assert body["required_topics"]
    assert len(body["recommendations"]) >= 1
    top = body["recommendations"][0]
    assert {"book", "author", "score", "matched_topics", "missing_topics"} <= top.keys()
    assert top["book"]
    assert 0 <= top["score"] <= 1
    assert top["matched_topics"]
    assert body["explanation"]


def test_recommendation_author_flows_to_coverage() -> None:
    """The book author (Phase 12) is carried from the indexed payloads."""
    service = rag_factory.make_indexing_service(
        graph_service=make_graph_service(),
        sparse_index=make_sparse_index(),
    )
    rag_factory.index(
        service,
        rag_factory.sample_structured(SYLLABUS_TEXT, document_id="doc-syllabus"),
    )
    rag_factory.index(
        service,
        make_library_book(
            document_id="doc-book-strong",
            title="Database System Concepts",
            author="Abraham Silberschatz",
            chapters=BOOK_STRONG[1],
        ),
    )
    recommender = make_recommendation_service(
        repository=service.repository,
        embedder=service.embedder,
        graph_service=service.graph_service,
    )
    result = _run(recommender.recommend("Which book is best for normalization?"))
    assert result.recommendations
    top = result.recommendations[0]
    assert top.book_title == "Database System Concepts"
    assert top.author == "Abraham Silberschatz"


def test_recommendations_endpoint_absent_topic(client, make_text_pdf, tmp_path) -> None:
    _index_pdf(client, make_text_pdf, tmp_path, "syllabus.pdf", SYLLABUS_TEXT)
    response = client.post("/recommendations", json={"query": "quantum computing book"})
    assert response.status_code == 200
    body = response.json()
    assert body["no_syllabus_evidence"] is True
    assert body["recommendations"] == []


def test_recommendations_empty_query_rejected(client) -> None:
    response = client.post("/recommendations", json={"query": ""})
    assert response.status_code == 422


def test_recommendations_empty_index_is_graceful(client) -> None:
    """A totally empty index (no collections yet) must not 500 (Phase 12)."""
    response = client.post("/recommendations", json={"query": "Recommend books for Computer Networks"})
    assert response.status_code == 200
    body = response.json()
    assert body["no_syllabus_evidence"] is True
    assert body["recommendations"] == []
    assert body["explanation"]
