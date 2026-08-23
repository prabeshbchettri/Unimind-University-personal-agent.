"""OCR backends."""

from .base import OCRBackend
from .tesseract import TesseractOCRBackend, find_tesseract_binary

__all__ = ["OCRBackend", "TesseractOCRBackend", "find_tesseract_binary"]