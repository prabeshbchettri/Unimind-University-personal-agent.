"""PDF type analysis: text-readable vs scanned.

The analyzer probes each page with a cheap text extraction, measures how much
usable text exists, and decides which extraction path to use. It never relies
on the filename or file extension.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import settings


@dataclass
class PDFProfile:
    """Statistics measured from a probe extraction."""

    page_count: int
    page_char_counts: list[int]
    total_chars: int
    text_pages: int
    text_ratio: float


class PDFAnalyzer:
    """Decide the extraction method for a PDF from its raw page texts."""

    def __init__(
        self,
        min_chars_per_page: int | None = None,
        min_text_ratio: float | None = None,
        force_method: str | None = None,
    ) -> None:
        # Configuration falls back to application settings when not supplied,
        # so detection is tunable via environment variables.
        self.min_chars_per_page = (
            min_chars_per_page if min_chars_per_page is not None else settings.pdf_analyzer_min_chars_per_page
        )
        self.min_text_ratio = (
            min_text_ratio if min_text_ratio is not None else settings.pdf_analyzer_min_text_ratio
        )
        self.force_method = (
            force_method if force_method is not None else settings.pdf_analyzer_force_method
        )
        if self.force_method not in ("auto", "pymupdf", "ocr"):
            raise ValueError(
                f"pdf_analyzer_force_method must be 'auto', 'pymupdf' or 'ocr', got {self.force_method!r}"
            )

    def profile(self, page_texts: list[str]) -> PDFProfile:
        """Measure extraction quality for a list of per-page probe texts."""
        counts = [len((text or "").strip()) for text in page_texts]
        page_count = len(counts)
        text_pages = sum(1 for count in counts if count >= self.min_chars_per_page)
        total_chars = sum(counts)
        text_ratio = text_pages / page_count if page_count else 0.0
        return PDFProfile(
            page_count=page_count,
            page_char_counts=counts,
            total_chars=total_chars,
            text_pages=text_pages,
            text_ratio=text_ratio,
        )

    def decide(self, page_texts: list[str]) -> str:
        """Return ``"pymupdf"`` or ``"ocr"`` for the given probe texts."""
        if self.force_method != "auto":
            return self.force_method

        if not page_texts:
            return "ocr"

        profile = self.profile(page_texts)
        if profile.text_ratio >= self.min_text_ratio:
            return "pymupdf"
        return "ocr"
