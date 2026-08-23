"""Tests for the ingestion pipeline (PyMuPDF path)."""

from pathlib import Path

import pytest

import app.ingestion.pipeline as pipeline_module
from app.ingestion import IngestionPipeline
from app.ingestion.models import PageResult


def test_text_pdf_uses_pymupdf_and_preserves_pages(text_pdf: Path) -> None:
    document = IngestionPipeline().ingest_file(text_pdf)
    assert document.extraction_method == "pymupdf"
    assert document.page_count == 2
    assert [p.page_number for p in document.pages] == [1, 2]
    assert "Database Management Systems" in document.pages[0].text


def test_document_id_and_filename(text_pdf: Path) -> None:
    document = IngestionPipeline().ingest_file(text_pdf)
    assert document.document_id
    assert document.filename == "syllabus_sample.pdf"
    assert document.metadata["source_file"] == "syllabus_sample.pdf"


def test_document_ids_are_unique(text_pdf: Path) -> None:
    pipeline = IngestionPipeline()
    first = pipeline.ingest_file(text_pdf)
    second = pipeline.ingest_file(text_pdf)
    assert first.document_id != second.document_id


def test_cleaning_is_applied(make_text_pdf, tmp_path: Path) -> None:
    noisy = make_text_pdf(
        tmp_path / "noisy.pdf",
        ["Unit  1    normalization   \r\n\n\n\n      second line"],
    )
    document = IngestionPipeline().ingest_file(noisy)
    assert document.pages[0].text == "Unit 1 normalization\nsecond line"


def test_page_error_does_not_destroy_document(text_pdf: Path, monkeypatch) -> None:
    def fake_extract_pages(self) -> list[PageResult]:
        return [
            PageResult(page_number=1, text="Page one content"),
            PageResult(page_number=2, error="boom"),
            PageResult(page_number=3, text="Page three content"),
        ]

    monkeypatch.setattr(
        pipeline_module.PyMuPDFExtractor, "extract_pages", fake_extract_pages
    )
    document = IngestionPipeline().ingest_file(text_pdf)

    assert document.page_count == 3
    assert document.pages[1].text == ""
    assert len(document.metadata["page_errors"]) == 1
    assert document.metadata["page_errors"][0]["page_number"] == 2
    assert "Page one content" in document.pages[0].text


def test_all_text_pages_extracted(text_pdf: Path) -> None:
    document = IngestionPipeline().ingest_file(text_pdf)
    assert all(page.text for page in document.pages)
    assert document.metadata["page_errors"] == []