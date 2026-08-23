"""Page extractors.

An extractor produces one :class:`PageResult` per page of an already-opened
PyMuPDF document. Individual page failures are captured as ``error`` values on
the result rather than being raised, so one bad page never destroys the whole
document.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pymupdf

from app.ingestion.models import PageResult


class PageExtractor(ABC):
    """Contract for extracting per-page text from an open PDF document."""

    def __init__(self, document: pymupdf.Document) -> None:
        self.document = document

    @abstractmethod
    def extract_pages(self) -> list[PageResult]:
        """Return one :class:`PageResult` for every page in the document."""
        raise NotImplementedError