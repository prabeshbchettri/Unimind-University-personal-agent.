"""Library recommendation evaluation (Phase 13).

Each scenario runs in its own hermetic mini-corpus so the coverage behavior
is exercised independently:

- high coverage   — one book covers every required topic
- medium coverage — the book covers some required topics
- low coverage    — the book covers almost nothing required
- missing topic   — a required topic is absent from a book's profile
- multiple similar books — two books cover the same topics; ranking must
  follow actual topic coverage

Every check verifies that the ranking is *supported by the indexed topic
coverage*: a book whose profile contains the required topic is ranked above
one that does not, and the score equals the documented formula
(0.85 * topic_coverage + 0.15 * subtopic_coverage).
"""

from __future__ import annotations

import asyncio

from app.chunking import SemanticChunker
from app.embedding import DeterministicEmbedder
from app.evaluation.corpus import _structured
from app.graph.extractor import GraphExtractor
from app.repositories.graph import InMemoryGraphRepository
from app.repositories.qdrant import QdrantRepository
from app.services.graph import KnowledgeGraphService
from app.services.recommendation import RecommendationService
from app.services.vector_indexing import VectorIndexingService
from app.structure.models import StructuredDocument

COLLECTIONS = ["university_docs", "past_questions", "library_books"]

# Syllabus topics used by the scenarios (subject expansion -> all required).
REQUIRED_TOPICS = ["Introduction", "Normalization", "Transactions", "Indexing"]
SUBTOPIC_MAP = {
    "Introduction": ["Data Models"],
    "Normalization": ["Normal Forms"],
    "Transactions": ["ACID Properties"],
    "Indexing": ["B-Trees"],
}


def _run(coro):
    return asyncio.run(coro)


def _syllabus(document_id: str = "doc-syllabus") -> StructuredDocument:
    lines = ["Database Management System Syllabus", ""]
    for index, topic in enumerate(REQUIRED_TOPICS, start=1):
        lines.append(f"Unit {index}: {topic}")
        for sub in SUBTOPIC_MAP[topic]:
            lines.append(f"{index}.1 {sub}")
        lines.append(f"The unit covers {topic}.")
        lines.append("")
    return _structured(
        document_id,
        "syllabus",
        ["\n".join(lines)],
        title="DBMS Syllabus",
        subject="Database Management System",
        semester=5,
        topics=REQUIRED_TOPICS,
        chapters=[(topic, SUBTOPIC_MAP[topic]) for topic in REQUIRED_TOPICS],
    )


def _book(document_id: str, title: str, topics: list[tuple[str, list[str]]], author: str = "Test Author") -> StructuredDocument:
    lines = []
    for index, (topic, subtopics) in enumerate(topics, start=1):
        lines.append(f"Chapter {index}: {topic}")
        for sub in subtopics:
            lines.append(f"{index}.1 {sub}")
        lines.append("")
    return _structured(
        document_id,
        "library_book",
        ["\n".join(lines)],
        title=title,
        author=author,
        topics=[topic for topic, _ in topics],
        chapters=topics,
    )


def _scenario_env(documents: list[StructuredDocument]) -> tuple[VectorIndexingService, RecommendationService]:
    embedder = DeterministicEmbedder(dimensions=384)
    repository = QdrantRepository(url=":memory:", collection_names=COLLECTIONS, default_dimensions=384)
    graph_service = KnowledgeGraphService(repository=InMemoryGraphRepository(), extractor=GraphExtractor())
    indexing = VectorIndexingService(
        embedder=embedder,
        repository=repository,
        collection_names=COLLECTIONS,
        chunker=SemanticChunker(max_chars=1200, overlap_chars=150),
        graph_service=graph_service,
    )
    for document in documents:
        _run(indexing.index(document))
    recommender = RecommendationService(
        repository=repository,
        embedder=embedder,
        graph_service=graph_service,
        llm=None,
        # Evaluation uses exact normalized-name matching only: the crude
        # deterministic embedder would otherwise match loosely-related
        # subtopics (e.g. "First Normal Form" vs "Normal Forms") above the
        # default 0.4 threshold. Semantic synonym matching is a real-model
        # behavior validated by the model-backed integration tests instead.
        semantic_threshold=1.0,
    )
    return indexing, recommender


def _query_books(recommender: RecommendationService, query: str) -> list[dict]:
    result = _run(recommender.recommend(query))
    return [
        {
            "book": coverage.book_title,
            "author": coverage.author,
            "score": coverage.score,
            "matched_topics": coverage.matched_topics,
            "missing_topics": coverage.missing_topics,
            "topic_coverage": coverage.topic_coverage,
            "subtopic_coverage": coverage.subtopic_coverage,
        }
        for coverage in result.recommendations
    ]


def evaluate_recommendations(env) -> dict:
    """Run the five recommendation scenarios on dedicated mini-corpora."""
    del env  # scenarios build their own hermetic environments

    # --- high / medium / low / missing-topic ---------------------------------
    _, strong_recommender = _scenario_env(
        [
            _syllabus("doc-syllabus"),
            _book("doc-strong", "Complete Database Systems", [(t, SUBTOPIC_MAP[t]) for t in REQUIRED_TOPICS]),
            _book("doc-partial", "Introduction to Databases", [("Introduction", ["Data Models"]), ("Normalization", ["Normal Forms"])]),
            _book("doc-poor", "Operating System Principles", [("Processes", ["Scheduling"]), ("Memory", ["Paging"])]),
        ]
    )
    ranked = _query_books(strong_recommender, "Which book is best for normalization?")
    strong = next((b for b in ranked if b["book"] == "Complete Database Systems"), None)
    partial = next((b for b in ranked if b["book"] == "Introduction to Databases"), None)
    poor = next((b for b in ranked if b["book"] == "Operating System Principles"), None)
    required = {"Introduction", "Normalization", "Transactions", "Indexing"}

    checks = {
        "high_coverage_scores_highest": bool(strong and (partial is None or strong["score"] > partial["score"])),
        "high_coverage_matches_all_topics": bool(strong and set(strong["matched_topics"]) == required),
        "medium_below_high": bool(strong and partial and strong["score"] > partial["score"] > 0),
        "low_coverage_scores_zero": bool(poor and poor["score"] == 0.0 and poor["matched_topics"] == []),
        "missing_topic_reported": bool(
            strong
            and partial
            and "Normalization" in partial["matched_topics"]
            and "Transactions" in partial["missing_topics"]
        ),
        "score_follows_formula": bool(
            strong
            and strong["topic_coverage"] == 1.0
            and abs(strong["score"] - (0.85 * strong["topic_coverage"] + 0.15 * strong["subtopic_coverage"])) < 1e-6
        ),
    }

    # --- multiple similar books -------------------------------------------------
    # A single-topic syllabus with subtopics so the graph carries subtopic
    # nodes and the subtopic term of the coverage formula is exercised.
    _, similar_recommender = _scenario_env(
        [
            _structured(
                "doc-syllabus-2",
                "syllabus",
                [
                    "Database Management System Syllabus\n"
                    "Unit 1: Normalization\n"
                    "1.1 Normal Forms\n"
                    "1.2 First Normal Form\n"
                    "Normalization removes redundancy."
                ],
                title="DBMS Syllabus",
                subject="Database Management System",
                semester=5,
                topics=["Normalization"],
                subtopics=["Normal Forms", "First Normal Form"],
                chapters=[("Normalization", ["Normal Forms", "First Normal Form"])],
            ),
            _book("doc-a", "Database Design A", [("Normalization", ["Normal Forms"])]),
            _book("doc-b", "Database Design B", [("Normalization", ["Normal Forms", "First Normal Form"])]),
        ]
    )
    similar = _query_books(similar_recommender, "Which book explains normalization?")
    book_a = next((b for b in similar if b["book"] == "Database Design A"), None)
    book_b = next((b for b in similar if b["book"] == "Database Design B"), None)
    checks["similar_books_both_ranked"] = bool(book_a and book_b)
    checks["similar_books_richer_subtopics_first"] = bool(
        book_a and book_b and book_b["subtopic_coverage"] > book_a["subtopic_coverage"] and book_b["score"] > book_a["score"]
    )
    checks["similar_books_formula_matches"] = bool(
        book_a and book_b and abs(book_b["score"] - (0.85 * book_b["topic_coverage"] + 0.15 * book_b["subtopic_coverage"])) < 1e-6
    )

    return {
        "scenarios": {
            "high_coverage": {"query": "Which book is best for normalization?", "books": ranked},
            "multiple_similar_books": {
                "query": "Which book explains normalization?",
                "books": similar,
            },
        },
        "checks": checks,
        "checks_passed": all(checks.values()),
        "formula": "score = 0.85 * topic_coverage + 0.15 * subtopic_coverage",
    }