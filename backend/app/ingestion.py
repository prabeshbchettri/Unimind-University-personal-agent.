"""Document ingestion primitives: extraction, cleaning, chunking, metadata.

This module is intentionally free of network calls and external services so
that the text pipeline is easy to test and to reason about.

Pipeline position:  PDF -> extract pages -> clean -> chunk -> metadata chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


class DocumentIngestionError(RuntimeError):
    """Raised when a document cannot be ingested."""


@dataclass(frozen=True)
class Chunk:
    """A chunk of document text with its source metadata."""

    id: str
    document_name: str
    source_path: str
    page: int
    chunk_index: int
    text: str


def discover_pdfs(directory: Path) -> list[Path]:
    """Return all PDF files in ``directory``, sorted by name.

    The scan is non-recursive: the documents directory is flat by convention.
    """
    if not directory.is_dir():
        raise DocumentIngestionError(f"Documents directory not found: {directory}")
    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        raise DocumentIngestionError(f"No PDF files found in {directory}")
    return pdfs


def clean_text(text: str) -> str:
    """Normalize extracted PDF text.

    - unify line endings
    - re-join words hyphenated across line breaks ("atten-\ndance" -> "attendance")
    - join lines into a single space-separated text block

    Paragraph structure is deliberately not preserved: chunks are fed to an
    embedding model, which does not benefit from line breaks.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_size: int = 900, chunk_overlap: int = 150) -> list[str]:
    """Split ``text`` into overlapping chunks of at most ``chunk_size`` characters.

    Words are never split in the middle; chunks break on word boundaries. Each
    chunk after the first starts with the trailing words of the previous chunk
    (up to ``chunk_overlap`` characters) so context is preserved across the
    boundary. The function is deterministic: the same input always produces the
    same chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    words = text.split()
    chunks: list[str] = []
    current: list[str] = []

    def length(parts: list[str]) -> int:
        return len(" ".join(parts))

    for word in words:
        candidate = [*current, word]
        if current and length(candidate) > chunk_size:
            chunks.append(" ".join(current))

            # Start the next chunk with an overlap tail of the previous one.
            tail: list[str] = []
            for prev in reversed(current):
                if length([*tail, prev]) > chunk_overlap:
                    break
                tail.insert(0, prev)
            current = tail
            candidate = [*current, word]

        current = candidate

    if current:
        chunks.append(" ".join(current))
    return chunks


def extract_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract ``(page_number, text)`` for every page with extractable text.

    Page numbers are 1-based. Pages that contain no text (for example blank
    separator pages) are skipped. A PDF where *no* page yields text is treated
    as a scanned document and raises, because OCR is not implemented yet.
    """
    try:
        document = fitz.open(pdf_path)
    except Exception as exc:  # PyMuPDF raises various exception types
        raise DocumentIngestionError(f"Cannot open PDF {pdf_path.name}: {exc}") from exc

    pages: list[tuple[int, str]] = []
    try:
        with document:
            for position, page in enumerate(document, start=1):
                text = clean_text(page.get_text("text"))
                if text:
                    pages.append((position, text))
    except DocumentIngestionError:
        raise
    except Exception as exc:
        raise DocumentIngestionError(
            f"Failed to extract text from {pdf_path.name}: {exc}"
        ) from exc

    if not pages:
        raise DocumentIngestionError(
            f"{pdf_path.name}: no extractable text on any page "
            "(scanned PDF? OCR is not implemented yet)"
        )
    return pages


def chunk_pdf(pdf_path: Path, chunk_size: int = 900, chunk_overlap: int = 150) -> list[Chunk]:
    """Extract, clean and chunk one PDF, preserving source metadata."""
    chunks: list[Chunk] = []
    for page_number, page_text in extract_pdf_pages(pdf_path):
        for index, piece in enumerate(chunk_text(page_text, chunk_size, chunk_overlap)):
            chunks.append(
                Chunk(
                    id=f"{pdf_path.stem}::p{page_number}::c{index}",
                    document_name=pdf_path.name,
                    source_path=str(pdf_path),
                    page=page_number,
                    chunk_index=index,
                    text=piece,
                )
            )
    if not chunks:
        raise DocumentIngestionError(f"{pdf_path.name}: produced no chunks")
    return chunks
