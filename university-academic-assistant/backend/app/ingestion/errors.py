"""Exceptions raised by the document ingestion pipeline.

A single root exception (:class:`IngestionError`) lets callers catch any
ingestion failure; concrete subclasses allow precise handling.
"""

from __future__ import annotations


class IngestionError(Exception):
    """Base class for all document ingestion errors."""


class UnsupportedFileTypeError(IngestionError):
    """The supplied file is not a supported input type (e.g. not a PDF)."""


class InvalidPdfError(IngestionError):
    """The file does not contain valid PDF content."""


class PdfOpenError(IngestionError):
    """The PDF could not be opened (missing, locked or corrupt)."""


class EmptyPdfError(IngestionError):
    """The PDF opened successfully but contains no pages."""


class PageExtractionError(IngestionError):
    """A single page could not be extracted (recorded, not fatal)."""


class OcrError(IngestionError):
    """OCR processing failed."""


class OcrUnavailableError(OcrError):
    """The configured OCR backend is not available on this system."""
