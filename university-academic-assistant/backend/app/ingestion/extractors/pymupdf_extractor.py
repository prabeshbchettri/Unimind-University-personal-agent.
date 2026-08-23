"""PyMuPDF text extractor for text-readable PDFs."""

from __future__ import annotations

from app.ingestion.models import PageResult

from .base import PageExtractor


class PyMuPDFExtractor(PageExtractor):
    """Extract embedded text while preserving page boundaries."""

    def extract_pages(self) -> list[PageResult]:
        results: list[PageResult] = []
        for page_number in range(self.document.page_count):
            try:
                text = self.document.load_page(page_number).get_text("text")
                results.append(PageResult(page_number=page_number + 1, text=text))
            except Exception as exc:  # capture, do not abort the document
                results.append(
                    PageResult(
                        page_number=page_number + 1,
                        error=f"Page text extraction failed: {exc}",
                    )
                )
        return results