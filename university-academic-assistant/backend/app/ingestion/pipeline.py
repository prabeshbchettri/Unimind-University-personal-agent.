"""PDF ingestion pipeline.

Orchestrates validation, type analysis, extraction and cleaning, producing a
:class:`NormalizedDocument` regardless of whether the source PDF is text-based
or scanned.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pymupdf

from app.core.logging import get_logger
from app.ingestion.analyzer import PDFAnalyzer
from app.ingestion.cleaning import TextCleaner
from app.ingestion.errors import (
    EmptyPdfError,
    InvalidPdfError,
    OcrUnavailableError,
    PdfOpenError,
    UnsupportedFileTypeError,
)
from app.ingestion.extractors import OCRPDFExtractor, PyMuPDFExtractor
from app.ingestion.models import NormalizedDocument, PageContent, PageResult
from app.ingestion.ocr import OCRBackend, TesseractOCRBackend

logger = get_logger("app.ingestion.pipeline")

# PDFs must start with the PDF magic bytes regardless of their file extension.
_PDF_MAGIC = b"%PDF"


class IngestionPipeline:
    """High-level entry point for converting PDF files into normalized documents."""

    def __init__(
        self,
        analyzer: PDFAnalyzer | None = None,
        cleaner: TextCleaner | None = None,
        ocr_backend: OCRBackend | None = None,
    ) -> None:
        self.analyzer = analyzer or PDFAnalyzer()
        self.cleaner = cleaner or TextCleaner()
        self.ocr_backend = ocr_backend or TesseractOCRBackend()

    # -- public API ----------------------------------------------------------

    def ingest_file(self, path: str | Path) -> NormalizedDocument:
        """Ingest a PDF file and return a normalized document."""
        file_path = Path(path)
        self._validate_file(file_path)

        document = self._open_document(file_path)
        try:
            probe_texts = self._probe_pages(document)
            method = self.analyzer.decide(probe_texts)
            if method == "ocr" and not self.ocr_backend.is_available():
                raise OcrUnavailableError(
                    "OCR was selected but the Tesseract backend is unavailable."
                )

            logger.info(
                "Extraction method chosen",
                extra={
                    "extra_fields": {
                        "filename": file_path.name,
                        "method": method,
                        "page_count": document.page_count,
                    }
                },
            )

            extractor = (
                PyMuPDFExtractor(document)
                if method == "pymupdf"
                else OCRPDFExtractor(document, self.ocr_backend)
            )
            page_results = extractor.extract_pages()
            return self._build_document(file_path, method, page_results)
        finally:
            document.close()

    # -- validation ----------------------------------------------------------

    def _validate_file(self, file_path: Path) -> None:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.is_file():
            raise UnsupportedFileTypeError(f"Not a regular file: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{file_path.suffix or '(none)'}'. Only PDF files are supported."
            )

        with open(file_path, "rb") as handle:
            header = handle.read(len(_PDF_MAGIC))
        if not header.startswith(_PDF_MAGIC):
            raise InvalidPdfError(f"File is not a valid PDF: {file_path.name}")

    def _open_document(self, file_path: Path) -> pymupdf.Document:
        try:
            document = pymupdf.open(file_path)
        except Exception as exc:
            raise PdfOpenError(f"Could not open PDF '{file_path.name}': {exc}") from exc

        if document.page_count == 0:
            document.close()
            raise EmptyPdfError(f"PDF contains no pages: {file_path.name}")
        return document

    # -- extraction helpers --------------------------------------------------

    def _probe_pages(self, document: pymupdf.Document) -> list[str]:
        """Cheap per-page text probe used for type analysis."""
        texts: list[str] = []
        for page_number in range(document.page_count):
            try:
                texts.append(document.load_page(page_number).get_text("text"))
            except Exception:
                texts.append("")
        return texts

    def _build_document(
        self,
        file_path: Path,
        method: str,
        page_results: list[PageResult],
    ) -> NormalizedDocument:
        pages: list[PageContent] = []
        page_errors: list[dict] = []

        for result in page_results:
            if result.error:
                page_errors.append(
                    {"page_number": result.page_number, "error": result.error}
                )
                pages.append(PageContent(page_number=result.page_number, text=""))
                continue

            cleaned = self.cleaner.clean_page(result.text)
            pages.append(PageContent(page_number=result.page_number, text=cleaned))

        metadata = {
            "page_errors": page_errors,
            "source_file": file_path.name,
        }

        document = NormalizedDocument(
            document_id=str(uuid.uuid4()),
            filename=file_path.name,
            page_count=len(pages),
            extraction_method=method,
            pages=pages,
            metadata=metadata,
        )

        if page_errors:
            logger.warning(
                "Pages with extraction errors",
                extra={
                    "extra_fields": {
                        "document_id": document.document_id,
                        "error_count": len(page_errors),
                    }
                },
            )

        return document