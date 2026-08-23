"""Test helpers for building normalized documents without PDF files."""

from __future__ import annotations

from app.ingestion.models import NormalizedDocument, PageContent
from app.structure.models import (
    DocumentMetadata,
    DocumentStructure,
    StructuredDocument,
)


def make_normalized(
    page_texts: list[str],
    filename: str = "document.pdf",
    document_id: str = "doc-1",
    extraction_method: str = "pymupdf",
) -> NormalizedDocument:
    """Build a :class:`NormalizedDocument` directly from page texts."""
    return NormalizedDocument(
        document_id=document_id,
        filename=filename,
        page_count=len(page_texts),
        extraction_method=extraction_method,
        pages=[
            PageContent(page_number=index + 1, text=text)
            for index, text in enumerate(page_texts)
        ],
    )


def make_structured(
    page_texts: list[str],
    filename: str = "document.pdf",
    document_id: str = "doc-1",
    document_type: str = "syllabus",
    metadata: DocumentMetadata | None = None,
    structure: DocumentStructure | None = None,
) -> StructuredDocument:
    """Build a :class:`StructuredDocument` directly from page texts."""
    pages = [
        PageContent(page_number=index + 1, text=text)
        for index, text in enumerate(page_texts)
    ]
    return StructuredDocument(
        document_id=document_id,
        filename=filename,
        extraction_method="pymupdf",
        page_count=len(pages),
        pages=pages,
        metadata=metadata or DocumentMetadata(document_type=document_type),
        structure=structure or DocumentStructure(),
    )
