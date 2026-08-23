"""Tests for ingestion error handling."""

from pathlib import Path

import pytest

from app.ingestion import (
    EmptyPdfError,
    IngestionPipeline,
    InvalidPdfError,
    PdfOpenError,
    UnsupportedFileTypeError,
)


def test_unsupported_extension(tmp_path: Path) -> None:
    file = tmp_path / "notice.txt"
    file.write_text("hello")
    with pytest.raises(UnsupportedFileTypeError):
        IngestionPipeline().ingest_file(file)


def test_non_pdf_content_is_rejected(tmp_path: Path) -> None:
    file = tmp_path / "fake.pdf"
    file.write_bytes(b"this is definitely not a pdf")
    with pytest.raises(InvalidPdfError):
        IngestionPipeline().ingest_file(file)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        IngestionPipeline().ingest_file(tmp_path / "missing.pdf")


def test_empty_pdf_raises(tmp_path: Path) -> None:
    # A minimal, valid PDF containing zero pages.
    empty_pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [ ] /Count 0 >>\nendobj\n"
        b"trailer\n<< /Size 2 /Root 1 0 R >>\n"
        b"startxref\n90\n%%EOF\n"
    )
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(empty_pdf)
    with pytest.raises(EmptyPdfError):
        IngestionPipeline().ingest_file(empty)


def test_corrupt_pdf_raises(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"%PDF-1.4\n% truncated corrupt payload")
    with pytest.raises(PdfOpenError):
        IngestionPipeline().ingest_file(corrupt)