"""Normalized internal representation of an ingested document.

Both extraction pipelines (PyMuPDF and OCR) produce the exact same structure,
so downstream components never need to know how the text was obtained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

ExtractionMethod = Literal["pymupdf", "ocr", "mixed"]


@dataclass
class PageResult:
    """Raw result of extracting a single page."""

    page_number: int
    text: str = ""
    error: Optional[str] = None


@dataclass
class PageContent:
    """Cleaned text content of a single page."""

    page_number: int
    text: str


@dataclass
class NormalizedDocument:
    """Common output of the ingestion pipeline."""

    document_id: str
    filename: str
    page_count: int
    extraction_method: ExtractionMethod
    pages: list[PageContent] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to the stable wire format used across the system."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "page_count": self.page_count,
            "extraction_method": self.extraction_method,
            "pages": [
                {"page_number": page.page_number, "text": page.text}
                for page in self.pages
            ],
            "metadata": self.metadata,
        }

    def total_chars(self) -> int:
        """Return the total number of extracted characters."""
        return sum(len(page.text) for page in self.pages)
