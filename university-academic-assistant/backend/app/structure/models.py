"""Schema-validated structured document models (Phase 3).

Every field is optional (except ``document_type``) so missing metadata is
represented as ``None`` rather than fabricated values. Later phases (chunking,
Qdrant metadata, Neo4j entities, library recommendation) consume these models.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.ingestion.models import PageContent

DocumentType = Literal[
    "syllabus",
    "notice",
    "past_question",
    "rules_regulations",
    "academic_calendar",
    "library_book",
    "unknown",
]


class DocumentMetadata(BaseModel):
    """Common metadata extracted from a document's content."""

    document_type: DocumentType
    title: str | None = None
    university: str | None = None
    program: str | None = None
    semester: int | None = None
    subject: str | None = None
    subject_code: str | None = None
    academic_year: str | None = None
    year: int | None = None
    question_number: int | None = None
    marks: int | None = None
    topic: str | None = None
    subtopic: str | None = None
    author: str | None = None


class Question(BaseModel):
    """A single question extracted from a past question paper."""

    question_number: int | None = None
    text: str
    marks: int | None = None


class Chapter(BaseModel):
    """A chapter (with its topics/subtopics) from a library book."""

    chapter_number: int | None = None
    title: str
    topics: list[str] = Field(default_factory=list)
    subtopics: list[str] = Field(default_factory=list)


class DocumentStructure(BaseModel):
    """Type-specific structure extracted from the document."""

    topics: list[str] = Field(default_factory=list)
    subtopics: list[str] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
    chapters: list[Chapter] = Field(default_factory=list)


class StructuredDocument(BaseModel):
    """Validated output of the structure pipeline.

    Combines the normalized extraction (pages) with classification, metadata
    and structure. Suitable for chunking, vector-store payloads, graph entity
    extraction and library recommendation.
    """

    document_id: str
    filename: str
    extraction_method: str
    page_count: int
    pages: list[PageContent] = Field(default_factory=list)
    metadata: DocumentMetadata
    structure: DocumentStructure = Field(default_factory=DocumentStructure)

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return self.model_dump()

    def metadata_payload(self) -> dict:
        """Metadata dict suitable for vector-store / graph payloads."""
        return {
            "document_id": self.document_id,
            "document_type": self.metadata.document_type,
            "title": self.metadata.title,
            "university": self.metadata.university,
            "program": self.metadata.program,
            "semester": self.metadata.semester,
            "subject": self.metadata.subject,
            "subject_code": self.metadata.subject_code,
            "academic_year": self.metadata.academic_year,
            "year": self.metadata.year,
            "question_number": self.metadata.question_number,
            "marks": self.metadata.marks,
            "topic": self.metadata.topic,
            "subtopic": self.metadata.subtopic,
            "author": self.metadata.author,
        }
