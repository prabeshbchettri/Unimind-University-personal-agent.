"""Recommendation service (Phase 9).

Evidence-based library book recommendation:

    User query
      -> TopicResolver: required syllabus topics (vector evidence + graph)
      -> Book profiles from the indexed ``library_books`` chunk payloads
      -> CoverageCalculator: transparent per-book coverage score
      -> rank() descending -> grounded explanation (LLM, never inventing)

Every score comes from the indexed content; the LLM only explains it.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from app.llm.base import LLMClient
from app.recommendation.coverage import CoverageCalculator, rank
from app.recommendation.models import BookCoverage, BookProfile, RecommendationResult
from app.recommendation.resolver import TopicResolver
from app.services.base import Service

if TYPE_CHECKING:
    from app.repositories.qdrant import QdrantRepository

_EXPLAIN_SYSTEM_PROMPT = (
    "You explain a library book recommendation for a university student. "
    "Rules:\n"
    "- Use ONLY the evidence provided (per-book score, matched topics, missing topics).\n"
    "- NEVER invent book contents, chapters, authors, or scores.\n"
    "- State what the top book covers, what it misses, and why it ranks first.\n"
    "Reply with 2-4 sentences."
)


class RecommendationService(Service):
    """Recommends library books grounded in syllabus/topic coverage."""

    def __init__(
        self,
        repository: QdrantRepository,
        embedder,
        graph_service=None,
        llm: LLMClient | None = None,
        semantic_threshold: float = 0.4,
        topic_weight: float = 0.85,
        subtopic_weight: float = 0.15,
        top_k: int = 10,
    ) -> None:
        super().__init__()
        self.repository = repository
        self.resolver = TopicResolver(
            embedder=embedder,
            repository=repository,
            graph_service=graph_service,
            top_k=top_k,
        )
        self.calculator = CoverageCalculator(
            embedder=embedder,
            semantic_threshold=semantic_threshold,
            topic_weight=topic_weight,
            subtopic_weight=subtopic_weight,
        )
        self.llm = llm

    async def health(self) -> dict:
        return {
            "name": self.name,
            "status": "ok",
            "calculator": (
                f"coverage = {self.calculator.topic_weight} * topic + "
                f"{self.calculator.subtopic_weight} * subtopic"
            ),
            "semantic_threshold": self.calculator.semantic_threshold,
        }

    async def recommend(self, query: str) -> RecommendationResult:
        """Return ranked book recommendations for ``query``."""
        context = await self.resolver.identify(query)

        if context.no_syllabus_evidence or not context.required_topics:
            return RecommendationResult(
                query=query,
                subject=context.subject,
                required_topics=context.required_topics,
                required_subtopics=context.required_subtopics,
                explanation="No syllabus evidence matched the query, so no books can be recommended.",
                no_syllabus_evidence=True,
            )

        books = await self._book_profiles()
        coverages = [
            self.calculator.calculate(book, context.required_topics, context.required_subtopics)
            for book in books
        ]
        recommendations = rank(coverages)
        explanation = await self._explain(query, context.subject, context.required_topics, recommendations)

        return RecommendationResult(
            query=query,
            subject=context.subject,
            required_topics=context.required_topics,
            required_subtopics=context.required_subtopics,
            recommendations=recommendations,
            explanation=explanation,
        )

    async def _book_profiles(self) -> list[BookProfile]:
        """Build one profile per indexed library book from its chunk payloads."""
        payloads = await self.repository.list_points("library_books")
        by_book: dict[str, dict] = {}
        for payload in payloads:
            if payload.get("document_type") != "library_book":
                continue
            document_id = payload.get("document_id")
            if not document_id:
                continue
            profile = by_book.setdefault(
                document_id,
                {
                    "title": payload.get("title") or document_id,
                    "document_id": document_id,
                    "author": payload.get("author"),
                    "topics": [],
                    "subtopics": [],
                },
            )
            topic = payload.get("topic")
            if topic and topic not in profile["topics"]:
                profile["topics"].append(topic)
            subtopic = payload.get("subtopic")
            if subtopic and subtopic not in profile["subtopics"]:
                profile["subtopics"].append(subtopic)
            for subtopic in payload.get("subtopics") or []:
                if subtopic and subtopic not in profile["subtopics"]:
                    profile["subtopics"].append(subtopic)
        return [BookProfile(**profile) for profile in by_book.values()]

    async def _explain(
        self,
        query: str,
        subject: str | None,
        required_topics: list[str],
        recommendations: list[BookCoverage],
    ) -> str:
        summary = _deterministic_summary(query, subject, required_topics, recommendations)
        if self.llm is None or not recommendations:
            return summary
        evidence = {
            "query": query,
            "subject": subject,
            "required_topics": required_topics,
            "recommendations": [
                {
                    "book": coverage.book_title,
                    "score": coverage.score,
                    "matched_topics": coverage.matched_topics,
                    "missing_topics": coverage.missing_topics,
                }
                for coverage in recommendations
            ],
        }
        try:
            return self.llm.complete(_EXPLAIN_SYSTEM_PROMPT, json.dumps(evidence, ensure_ascii=False))
        except Exception:  # noqa: BLE001 - fall back to the deterministic summary
            return summary


def _deterministic_summary(
    query: str,
    subject: str | None,
    required_topics: list[str],
    recommendations: list[BookCoverage],
) -> str:
    if not recommendations:
        return "No library books are indexed, so nothing can be recommended."
    top = recommendations[0]
    lines = [
        f"{top.book_title} covers {len(top.matched_topics)} of {len(required_topics)} "
        f"required topic(s) ({top.score:.0%})"
    ]
    if top.matched_topics:
        lines.append(f"Matched topics: {', '.join(top.matched_topics)}.")
    if top.missing_topics:
        lines.append(f"Missing topics: {', '.join(top.missing_topics)}.")
    if len(recommendations) > 1:
        runner = recommendations[1]
        lines.append(f"Next best: {runner.book_title} ({runner.score:.0%}).")
    return " ".join(lines)
