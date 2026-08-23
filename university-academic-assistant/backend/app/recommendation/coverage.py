"""Transparent topic-coverage scoring (Phase 9).

The score is computed from retrieved evidence only and its formula is
deliberately simple and documented:

    coverage_score = topic_weight * topic_coverage
                   + subtopic_weight * subtopic_coverage

where

    topic_coverage     = |matched required topics| / |required topics|
    subtopic_coverage  = |matched required subtopics| / |required subtopics|
                         (0.0 when there are no required subtopics)

A required topic matches a book topic when the normalized names are equal
(exact match), or when their embedding cosine similarity is at or above
``semantic_threshold`` (semantic match — catches synonyms and paraphrases).
The weights (default topic 0.85, subtopic 0.15) prefer full-topic coverage
while still rewarding books that reach into the required subtopics; both
values are explicit so the ranking can be audited per book.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.recommendation.models import BookCoverage, BookProfile


def normalize_name(name: str) -> str:
    """Lowercase, collapse whitespace and drop punctuation for name matching."""
    value = re.sub(r"[^\w\s]", " ", name or "").lower()
    return re.sub(r"\s+", " ", value).strip()


class CoverageCalculator:
    """Computes how well a book covers the required syllabus topics."""

    def __init__(
        self,
        embedder=None,
        semantic_threshold: float = 0.4,
        topic_weight: float = 0.85,
        subtopic_weight: float = 0.15,
    ) -> None:
        self.embedder = embedder
        self.semantic_threshold = semantic_threshold
        self.topic_weight = topic_weight
        self.subtopic_weight = subtopic_weight

    # -- matching -----------------------------------------------------------

    def _matches(self, required: str, candidates: Sequence[str]) -> bool:
        required_norm = normalize_name(required)
        for candidate in candidates:
            if normalize_name(candidate) == required_norm:
                return True
        if self.embedder is not None:
            required_vec = self.embedder.embed_one(required)
            for candidate in candidates:
                candidate_vec = self.embedder.embed_one(candidate)
                similarity = _cosine(required_vec, candidate_vec)
                if similarity >= self.semantic_threshold:
                    return True
        return False

    def _matched(self, required: Sequence[str], candidates: Sequence[str]) -> list[str]:
        return [name for name in required if self._matches(name, candidates)]

    # -- scoring ------------------------------------------------------------

    def calculate(
        self,
        book: BookProfile,
        required_topics: Sequence[str],
        required_subtopics: Sequence[str],
    ) -> BookCoverage:
        """Coverage of ``book`` against the required topics (transparent)."""
        matched_topics = self._matched(required_topics, book.topics)
        missing_topics = [topic for topic in required_topics if topic not in matched_topics]

        matched_subtopics = self._matched(required_subtopics, book.subtopics)
        missing_subtopics = [sub for sub in required_subtopics if sub not in matched_subtopics]

        topic_coverage = len(matched_topics) / len(required_topics) if required_topics else 0.0
        subtopic_coverage = len(matched_subtopics) / len(required_subtopics) if required_subtopics else 0.0
        score = self.topic_weight * topic_coverage + self.subtopic_weight * subtopic_coverage

        return BookCoverage(
            book_title=book.title,
            document_id=book.document_id,
            author=book.author,
            score=round(score, 4),
            topic_coverage=round(topic_coverage, 4),
            subtopic_coverage=round(subtopic_coverage, 4),
            matched_topics=matched_topics,
            missing_topics=missing_topics,
            matched_subtopics=matched_subtopics,
            missing_subtopics=missing_subtopics,
            chapter_count=book.chapter_count,
            evidence=[book.title, book.document_id],
        )


def rank(coverages: Sequence[BookCoverage]) -> list[BookCoverage]:
    """Rank books by coverage: score desc, then matched topics, then title asc."""
    return sorted(
        coverages,
        key=lambda coverage: (-coverage.score, -len(coverage.matched_topics), coverage.book_title),
    )


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
