"""Tests for the OCR backend and the OCR extraction path.

The OCR end-to-end test renders text into an image and embeds it in a PDF to
simulate a scanned document; it is skipped automatically when Tesseract is not
available on the machine.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest
from PIL import Image, ImageDraw, ImageFont

from app.ingestion import IngestionPipeline
from app.ingestion.errors import OcrUnavailableError
from app.ingestion.ocr import TesseractOCRBackend, find_tesseract_binary

_FONT_CANDIDATES = (
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def test_binary_is_resolved() -> None:
    binary = find_tesseract_binary()
    # Either found on this machine or, on machines without it, resolved to None.
    assert binary is None or Path(binary).exists()


def test_backend_availability_matches_binary() -> None:
    backend = TesseractOCRBackend()
    assert backend.is_available() == (backend.binary_path is not None)


def test_recognize_image_mocked(monkeypatch) -> None:
    backend = TesseractOCRBackend()
    image = Image.new("RGB", (10, 10), "white")
    monkeypatch.setattr(
        "pytesseract.image_to_string", lambda image_, lang=None: "MOCKED OCR TEXT"
    )
    assert backend.recognize_image(image) == "MOCKED OCR TEXT"


def test_unavailable_backend_raises() -> None:
    backend = TesseractOCRBackend(binary_path=r"C:\nonexistent\tesseract.exe")
    assert not backend.is_available()
    with pytest.raises(OcrUnavailableError):
        backend.recognize_image(Image.new("RGB", (10, 10), "white"))


def _render_text_image(text: str, path: Path) -> None:
    image = Image.new("RGB", (1500, 320), "white")
    draw = ImageDraw.Draw(image)
    font = None
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            font = ImageFont.truetype(candidate, 44)
            break
    if font is None:
        font = ImageFont.load_default()
    draw.text((40, 80), text, fill="black", font=font)
    image.save(path)


def _make_scanned_pdf(path: Path, page_texts: list[str]) -> Path:
    doc = pymupdf.open()
    for index, text in enumerate(page_texts):
        image_path = path.parent / f"page_{index}.png"
        _render_text_image(text, image_path)
        page = doc.new_page(width=595, height=842)
        page.insert_image(page.rect, filename=str(image_path))
    doc.save(str(path))
    doc.close()
    return path


@pytest.mark.skipif(
    not TesseractOCRBackend().is_available(), reason="Tesseract not installed"
)
def test_scanned_pdf_is_detected_and_ocr_processed(tmp_path: Path) -> None:
    scanned = _make_scanned_pdf(
        tmp_path / "scanned_notice.pdf",
        ["UNIVERSITY EXAMINATION NOTICE", "SEMESTER FIVE EXAMINATIONS BEGIN NEXT MONTH"],
    )
    document = IngestionPipeline().ingest_file(scanned)

    assert document.extraction_method == "ocr"
    assert document.page_count == 2
    combined = " ".join(page.text for page in document.pages).upper()
    assert "EXAMINATION" in combined
    assert "SEMESTER" in combined