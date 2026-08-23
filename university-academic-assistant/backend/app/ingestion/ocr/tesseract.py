"""Tesseract OCR backend.

Wraps pytesseract behind the :class:`OCRBackend` interface. The binary path is
resolved from configuration or auto-detected from common locations / PATH.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from PIL import Image

from app.config import settings
from app.ingestion.errors import OcrError, OcrUnavailableError

from .base import OCRBackend

_COMMON_TESSERACT_PATHS = (
    "C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
    "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/opt/homebrew/bin/tesseract",
)


def find_tesseract_binary(configured: str = "") -> str | None:
    """Locate the Tesseract executable.

    An explicitly configured path (argument or ``TESSERACT_PATH``) is
    authoritative: if provided but missing, ``None`` is returned rather than
    silently falling back. Otherwise the ``PATH`` lookup, then common install
    locations, are searched.
    """
    explicit = configured or os.environ.get("TESSERACT_PATH", "")
    if explicit:
        return explicit if Path(explicit).exists() else None

    from_shutil = shutil.which("tesseract")
    if from_shutil:
        return from_shutil

    for candidate in _COMMON_TESSERACT_PATHS:
        if Path(candidate).exists():
            return candidate
    return None


class TesseractOCRBackend(OCRBackend):
    """OCR backend implemented with Tesseract via pytesseract."""

    name = "tesseract"

    def __init__(self, binary_path: str | None = None, lang: str | None = None) -> None:
        self.binary_path = find_tesseract_binary(binary_path or "")
        if self.binary_path:
            try:
                import pytesseract

                pytesseract.pytesseract.tesseract_cmd = self.binary_path
            except ImportError:  # pragma: no cover - depends on install
                self.binary_path = None
        self.lang = lang or settings.ocr_language

    def is_available(self) -> bool:
        return bool(self.binary_path)

    def recognize_image(self, image: Image.Image) -> str:
        if not self.is_available():
            raise OcrUnavailableError(
                "Tesseract OCR is not available. Install Tesseract and/or set TESSERACT_PATH."
            )
        try:
            import pytesseract
        except ImportError as exc:  # pragma: no cover - depends on install
            raise OcrUnavailableError("pytesseract is not installed.") from exc

        # Keep rendering fast and accurate: grayscale typically improves results.
        gray = image.convert("L")
        try:
            return pytesseract.image_to_string(gray, lang=self.lang)
        except Exception as exc:
            raise OcrError(f"Tesseract OCR failed: {exc}") from exc