"""OCR backend interface.

Extraction code depends only on this interface, so the Tesseract backend can be
replaced by PaddleOCR (or anything else) later without touching the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image


class OCRBackend(ABC):
    """Contract for any OCR engine used to transcribe rendered page images."""

    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether this backend can run on the current system."""
        raise NotImplementedError

    @abstractmethod
    def recognize_image(self, image: Image.Image) -> str:
        """Transcribe a rendered page image into text."""
        raise NotImplementedError