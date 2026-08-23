"""Tests for the PDF type analyzer."""

import pytest

from app.ingestion import PDFAnalyzer

TEXT_PAGE = "The quick brown fox jumps over the lazy dog. " * 4  # > min chars
SCANNED_PAGE = ""  # empty page simulates a scanned image


def test_text_document_uses_pymupdf() -> None:
    analyzer = PDFAnalyzer()
    assert analyzer.decide([TEXT_PAGE, TEXT_PAGE, TEXT_PAGE]) == "pymupdf"


def test_scanned_document_uses_ocr() -> None:
    analyzer = PDFAnalyzer()
    assert analyzer.decide([SCANNED_PAGE, SCANNED_PAGE]) == "ocr"


def test_partial_text_falls_back_to_ocr() -> None:
    analyzer = PDFAnalyzer()
    assert analyzer.decide([TEXT_PAGE, SCANNED_PAGE, SCANNED_PAGE]) == "ocr"


def test_force_method_overrides_analysis() -> None:
    text_forced = PDFAnalyzer(force_method="ocr")
    assert text_forced.decide([TEXT_PAGE]) == "ocr"

    scan_forced = PDFAnalyzer(force_method="pymupdf")
    assert scan_forced.decide([SCANNED_PAGE]) == "pymupdf"


def test_empty_pages_default_to_ocr() -> None:
    assert PDFAnalyzer().decide([]) == "ocr"


def test_invalid_force_method_raises() -> None:
    with pytest.raises(ValueError):
        PDFAnalyzer(force_method="bogus")


def test_profile_metrics() -> None:
    analyzer = PDFAnalyzer(min_chars_per_page=20)
    profile = analyzer.profile([TEXT_PAGE, SCANNED_PAGE, TEXT_PAGE])
    assert profile.page_count == 3
    assert profile.text_pages == 2
    assert profile.text_ratio == pytest.approx(2 / 3)