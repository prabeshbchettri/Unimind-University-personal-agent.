"""OCR extractor for scanned/image-based PDFs.

Each PDF page is rendered to an image and transcribed by an OCR backend. The
backend is injected so the pipeline stays independent of the OCR engine used.
"""

from __future__ import annotations

from PIL import Image

from app.config import settings
from app.ingestion.models import PageResult
from app.ingestion.ocr import OCRBackend

from .base import PageExtractor


class OCRPDFExtractor(PageExtractor):
    """Render pages to images and transcribe them with an OCR backend."""

    def __init__(self, document, ocr_backend: OCRBackend, dpi: int | None = None) -> None:
        super().__init__(document)
        self.ocr_backend = ocr_backend
        self.dpi = dpi or settings.ocr_dpi

    def extract_pages(self) -> list[PageResult]:
        results: list[PageResult] = []
        for page_number in range(self.document.page_count):
            page = self.document.load_page(page_number)
            try:
                pixmap = page.get_pixmap(dpi=self.dpi, colorspace="rgb")
                image = Image.frombytes(
                    "RGB", (pixmap.width, pixmap.height), pixmap.samples
                )
                text = self.ocr_backend.recognize_image(image)
                results.append(PageResult(page_number=page_number + 1, text=text))
            except Exception as exc:  # capture, do not abort the document
                results.append(
                    PageResult(
                        page_number=page_number + 1,
                        error=f"OCR extraction failed: {exc}",
                    )
                )
        return results