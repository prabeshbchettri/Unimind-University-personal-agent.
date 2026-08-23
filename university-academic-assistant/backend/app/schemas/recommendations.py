"""Recommendation endpoint schemas (Phase 9)."""

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    """Request body for ``POST /recommendations``."""

    query: str = Field(..., min_length=1, description="Topic or syllabus question to recommend books for.")


class BookRecommendationSchema(BaseModel):
    """One ranked book recommendation."""

    book: str = Field(..., description="Book title.")
    author: str | None = Field(None, description="Book author, when available.")
    score: float = Field(..., description="Coverage score (0..1).")
    matched_topics: list[str] = Field(default_factory=list, description="Required topics the book covers.")
    missing_topics: list[str] = Field(default_factory=list, description="Required topics the book does not cover.")
    matched_subtopics: list[str] = Field(default_factory=list, description="Required subtopics the book covers.")
    missing_subtopics: list[str] = Field(default_factory=list, description="Required subtopics the book does not cover.")
    topic_coverage: float = Field(0.0, description="matched_required_topics / required_topics (0..1).")
    subtopic_coverage: float = Field(0.0, description="matched_required_subtopics / required_subtopics (0..1).")
    chapter_count: int = Field(0, description="Distinct chapters indexed for the book.")
    evidence: list[str] = Field(default_factory=list, description="Sources backing the coverage.")


class RecommendationResponse(BaseModel):
    """Response body for ``POST /recommendations``."""

    query: str = Field(..., description="The original query.")
    subject: str | None = Field(None, description="Syllabus subject the query maps to (if identifiable).")
    required_topics: list[str] = Field(default_factory=list, description="Required topics from the syllabus.")
    required_subtopics: list[str] = Field(default_factory=list, description="Required subtopics from the syllabus.")
    recommendations: list[BookRecommendationSchema] = Field(
        default_factory=list, description="Books ranked by coverage (descending)."
    )
    explanation: str = Field("", description="Grounded explanation of the ranking.")
    no_syllabus_evidence: bool = Field(False, description="True when no syllabus evidence matched the query.")
