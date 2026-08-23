"""Semantic chunk models."""

from __future__ import annotations

from pydantic import BaseModel


class DocumentChunk(BaseModel):
    """A self-contained, searchable unit of a structured document.

    Each chunk preserves the provenance of its content (document, page,
    section, topic, subtopic) and carries document-level metadata so payloads
    stored in the vector store are self-describing.
    """

    chunk_id: str
    document_id: str
    document_type: str
    title: str | None = None
    author: str | None = None
    page: int
    section: str | None = None
    topic: str | None = None
    subtopic: str | None = None
    semester: int | None = None
    subject: str | None = None
    subject_code: str | None = None
    academic_year: str | None = None
    marks: int | None = None
    question_number: int | None = None
    text: str

    def payload(self) -> dict:
        """Payload dict to store as the Qdrant point payload."""
        return self.model_dump()
