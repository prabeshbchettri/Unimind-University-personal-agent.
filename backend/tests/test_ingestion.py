"""Tests for extraction, cleaning, chunking and chunk metadata.

These tests create real (tiny) PDFs with PyMuPDF so extraction runs against
actual PDF files, not mocks.
"""

from __future__ import annotations

import pytest

from app.ingestion import (
    DocumentIngestionError,
    chunk_pdf,
    chunk_text,
    clean_text,
    discover_pdfs,
    extract_pdf_pages,
)
from tests.conftest import write_pdf

PAGE_A = (
    "The university requires a minimum attendance of seventy-five percent. "
    "Students below the threshold must apply for conditional permission."
)
PAGE_B = (
    "Examination eligibility requires registration in good standing and "
    "settlement of outstanding fees."
)


def test_clean_text_joins_lines_and_repairs_hyphenation() -> None:
    raw = "minimum atten-\ndance is 75 percent.\r\nSecond line\rthird"
    assert clean_text(raw) == "minimum attendance is 75 percent. Second line third"


def test_clean_text_collapses_whitespace() -> None:
    assert clean_text("a\n\n  b\t c  ") == "a b c"


def test_chunk_text_respects_size_and_word_boundaries() -> None:
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 100
        assert chunk == chunk.strip()
        assert not chunk.endswith("-")


def test_chunk_text_produces_overlap_between_neighbors() -> None:
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_text(text, chunk_size=80, chunk_overlap=25)
    for previous, current in zip(chunks, chunks[1:]):
        prev_words = previous.split()
        cur_words = current.split()
        shared = 0
        for k in range(1, min(len(prev_words), len(cur_words)) + 1):
            if prev_words[-k:] == cur_words[:k]:
                shared = k
        assert shared >= 1, "consecutive chunks must share overlapping words"


def test_chunk_text_single_short_text_returns_one_chunk() -> None:
    assert chunk_text("short text") == ["short text"]


def test_chunk_text_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size=10, chunk_overlap=10)


def test_extract_pdf_pages_numbers_pages_and_cleans_text(tmp_path) -> None:
    pdf = write_pdf(
        tmp_path / "two_pages.pdf",
        "Test Doc",
        [PAGE_A],  # first page
    )
    # Append a second page with different content.
    import fitz

    document = fitz.open(pdf)
    page = document.new_page(width=595, height=842)
    page.insert_text((60, 60), PAGE_B, fontname="helv", fontsize=11)
    document.save(pdf.with_name("two_pages_2.pdf"))
    document.close()

    pages = extract_pdf_pages(pdf.with_name("two_pages_2.pdf"))
    assert [number for number, _ in pages] == [1, 2]
    assert "seventy-five percent" in pages[0][1]
    assert "good standing" in pages[1][1]


def test_chunk_pdf_preserves_document_metadata(tmp_path) -> None:
    pdf = write_pdf(tmp_path / "attendance_policy.pdf", "Attendance", [PAGE_A])
    chunks = chunk_pdf(pdf, chunk_size=50, chunk_overlap=10)

    assert chunks, "chunking must produce at least one chunk"
    for index, chunk in enumerate(chunks):
        assert chunk.document_name == "attendance_policy.pdf"
        assert chunk.source_path == str(pdf)
        assert chunk.page == 1
        assert chunk.chunk_index == index
        assert chunk.id.startswith("attendance_policy::p1::c")
        assert chunk.text


def test_chunk_pdf_uses_page_numbers_in_ids(tmp_path) -> None:
    import fitz

    pdf = write_pdf(tmp_path / "multi.pdf", "Multi", [PAGE_A])
    two_page_pdf = pdf.with_name("multi_two_pages.pdf")
    document = fitz.open(pdf)
    page = document.new_page(width=595, height=842)
    page.insert_text((60, 60), PAGE_B, fontname="helv", fontsize=11)
    document.save(two_page_pdf)  # PyMuPDF cannot save over the open file
    document.close()

    chunks = chunk_pdf(two_page_pdf, chunk_size=40, chunk_overlap=8)
    assert {chunk.page for chunk in chunks} == {1, 2}
    assert any(chunk.id.startswith("multi_two_pages::p2::") for chunk in chunks)


def test_discover_pdfs_sorted_and_fails_on_empty_dir(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DocumentIngestionError):
        discover_pdfs(empty)

    write_pdf(tmp_path / "b.pdf", "B", ["text"])
    write_pdf(tmp_path / "a.pdf", "A", ["text"])
    assert [path.name for path in discover_pdfs(tmp_path)] == ["a.pdf", "b.pdf"]


def test_scanned_pdf_without_text_raises(tmp_path) -> None:
    import fitz

    pdf = tmp_path / "scanned.pdf"
    document = fitz.open()
    document.new_page(width=595, height=842)  # blank page, no text
    document.save(pdf)
    document.close()

    with pytest.raises(DocumentIngestionError, match="no extractable text"):
        extract_pdf_pages(pdf)
