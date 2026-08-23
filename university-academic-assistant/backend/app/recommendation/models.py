"""Book recommendation domain models (Phase 9).

Every score is computed from the indexed content — nothing is invented. A
``BookProfile`` is built from the payloads of the indexed library-book chunks
(topics = chapter titles, subtopics = numbered subsections), and
``BookCoverage`` reports exactly which required topics matched and which did
not, so the ranking is evidence-based.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class BookProfile(BaseModel):
    """Topics covered by one indexed library book (from its chunk payloads)."""

    title: str
    document_id: str
    author: str | None = Field(None, description="Book author, when the metadata extraction found one.")
    topics: list[str] = Field(default_factory=list, description="Distinct chapter titles (chunk topics).")
    subtopics: list[str] = Field(default_factory=list, description="Distinct numbered subsections (chunk subtopics).")

    @property
    def chapter_count(self) -> int:
        return len(self.topics)


class BookCoverage(BaseModel):
    """Coverage of the required syllabus topics by one book."""

    book_title: str
    document_id: str
    author: str | None = Field(None, description="Book author, when available.")
    # Overall score (0..1) — see coverage.py for the exact formula.
    score: float
    topic_coverage: float = Field(0.0, description="matched_required_topics / required_topics (0..1).")
    subtopic_coverage: float = Field(0.0, description="matched_required_subtopics / required_subtopics (0..1).")
    matched_topics: list[str] = Field(default_factory=list, description="Required topics the book covers.")
    missing_topics: list[str] = Field(default_factory=list, description="Required topics the book does not cover.")
    matched_subtopics: list[str] = Field(default_factory=list, description="Required subtopics the book covers.")
    missing_subtopics: list[str] = Field(default_factory=list, description="Required subtopics the book does not cover.")
    chapter_count: int = Field(0, description="Number of distinct chapters indexed for the book.")
    evidence: list[str] = Field(default_factory=list, description="Sources backing this coverage (title, document id).")


class RecommendationResult(BaseModel):
    """The full recommendation output for one query."""

    query: str
    subject: str | None = Field(None, description="Syllabus subject the query maps to (if identifiable).")
    required_topics: list[str] = Field(default_factory=list, description="Required topics from the university syllabus.")
    required_subtopics: list[str] = Field(default_factory=list, description="Required subtopics from the syllabus chunks.")
    recommendations: list[BookCoverage] = Field(default_factory=list, description="Books ranked by coverage (descending).")
    explanation: str = Field("", description="Grounded explanation of the ranking.")
    no_syllabus_evidence: bool = Field(False, description="True when no syllabus evidence matched the query.")
