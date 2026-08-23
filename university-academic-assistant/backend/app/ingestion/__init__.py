"""Document ingestion package.

Phase 2: PDF type detection, PyMuPDF text extraction, OCR (Tesseract) fallback,
conservative text cleaning and a normalized document representation.

Exposed public API:
- :class:`IngestionPipeline` — high-level ``ingest_file`` entry point.
- :class:`NormalizedDocument` — common output structure.
- Errors in :mod:`app.ingestion.errors`.
"""

from .analyzer import PDFAnalyzer
from .cleaning import TextCleaner
from .errors import (
    EmptyPdfError,
    IngestionError,
    InvalidPdfError,
    OcrError,
    OcrUnavailableError,
    PageExtractionError,
    PdfOpenError,
    UnsupportedFileTypeError,
)
from .models import NormalizedDocument, PageContent
from .ocr import OCRBackend, TesseractOCRBackend
from .pipeline import IngestionPipeline

__all__ = [
    "IngestionPipeline",
    "PDFAnalyzer",
    "TextCleaner",
    "NormalizedDocument",
    "PageContent",
    "OCRBackend",
    "TesseractOCRBackend",
    "IngestionError",
    "UnsupportedFileTypeError",
    "InvalidPdfError",
    "PdfOpenError",
    "EmptyPdfError",
    "PageExtractionError",
    "OcrError",
    "OcrUnavailableError",
]