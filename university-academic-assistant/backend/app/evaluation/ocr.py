"""OCR evaluation (Phase 13).

Compares the extraction pipeline on digital PDFs (text layer) and scanned
PDFs (image-only, OCR path) against the ground-truth text:

- **digital** — PyMuPDF text extraction; expected to be near-exact.
- **scanned** — Tesseract OCR of rendered pages; requires a Tesseract
  binary (``shutil.which("tesseract")``), otherwise the scenario is
  reported as skipped and the digital scenario still runs.

Metrics: character accuracy (normalized Levenshtein similarity) and exact
word-recall between the extracted page text and the ground truth.
"""

from __future__ import annotations

import asyncio
import difflib
import shutil
import tempfile
from pathlib import Path

from app.evaluation.environment import EvaluationEnvironment

DIGITAL_GROUND_TRUTH = (
    "Examination Regulations 2024\n"
    "Rule 8: Grading\n"
    "Marks are converted to grade points: 80-100 is A, 60-79 is B, "
    "40-59 is C, below 40 is F."
)

SCANNED_GROUND_TRUTH = (
    "Notice\n"
    "Attendance Requirement for 2026\n"
    "All students must attend at least 75 percent of classes."
)


def _run(coro):
    return asyncio.run(coro)


def _normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def _similarity(extracted: str, ground_truth: str) -> float:
    a, b = _normalize(extracted), _normalize(ground_truth)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _word_recall(extracted: str, ground_truth: str) -> float:
    expected = set(_normalize(ground_truth).split())
    actual = set(_normalize(extracted).split())
    if not expected:
        return 0.0
    return len(expected & actual) / len(expected)


def _make_digital_pdf(path: Path, text: str) -> None:
    import pymupdf

    document = pymupdf.open()
    page = document.new_page()
    page.insert_textbox((72, 72, 540, 760), text, fontsize=11)
    document.save(str(path))
    document.close()


def _make_scanned_pdf(path: Path, text: str) -> None:
    """Render the text to an image page (no text layer -> forces OCR)."""
    import pymupdf

    source = pymupdf.open()
    page = source.new_page()
    page.insert_textbox((72, 72, 540, 760), text, fontsize=11)
    pixmap = page.get_pixmap(dpi=150)
    image_bytes = pixmap.tobytes("png")

    target = pymupdf.open()
    image_page = target.new_page(width=pixmap.width, height=pixmap.height)
    image_page.insert_image(image_page.rect, stream=image_bytes)
    target.save(str(path))
    target.close()
    source.close()


def _extract(env: EvaluationEnvironment, path: Path) -> tuple[str, str]:
    document = _run(env.ingestion_service.ingest_document(str(path)))
    text = "\n".join(page.text for page in document.pages)
    return text, document.extraction_method


def evaluate_ocr(env: EvaluationEnvironment) -> dict:
    """Extract digital and scanned PDFs and report quality metrics."""
    tesseract = shutil.which("tesseract")
    report: dict = {
        "tesseract_available": tesseract is not None,
        "scenarios": {},
        "known_failure_cases": [
            "Low-resolution scans (below ~150 DPI) lose characters.",
            "Handwritten answers in past papers OCR poorly.",
            "Complex tables and multi-column layouts merge/split words.",
            "Small or decorative fonts may be misread as similar glyphs.",
            "Watermarks can be read as text and pollute extracted content.",
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        digital_pdf = Path(tmp) / "digital.pdf"
        _make_digital_pdf(digital_pdf, DIGITAL_GROUND_TRUTH)
        digital_text, digital_method = _extract(env, digital_pdf)
        report["scenarios"]["digital"] = {
            "extraction_method": digital_method,
            "character_accuracy": round(_similarity(digital_text, DIGITAL_GROUND_TRUTH), 4),
            "word_recall": round(_word_recall(digital_text, DIGITAL_GROUND_TRUTH), 4),
        }

        if tesseract:
            scanned_pdf = Path(tmp) / "scanned.pdf"
            _make_scanned_pdf(scanned_pdf, SCANNED_GROUND_TRUTH)
            scanned_text, scanned_method = _extract(env, scanned_pdf)
            report["scenarios"]["scanned"] = {
                "extraction_method": scanned_method,
                "character_accuracy": round(_similarity(scanned_text, SCANNED_GROUND_TRUTH), 4),
                "word_recall": round(_word_recall(scanned_text, SCANNED_GROUND_TRUTH), 4),
            }
        else:
            report["scenarios"]["scanned"] = {
                "extraction_method": "ocr",
                "status": "skipped",
                "reason": "Tesseract OCR binary not found on this machine.",
            }

    return report