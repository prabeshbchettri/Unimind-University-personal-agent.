"""Tests for the normalized document representation."""

from app.ingestion.models import NormalizedDocument, PageContent


def test_to_dict_matches_expected_structure() -> None:
    document = NormalizedDocument(
        document_id="doc-1",
        filename="syllabus.pdf",
        page_count=2,
        extraction_method="pymupdf",
        pages=[
            PageContent(page_number=1, text="Unit 1"),
            PageContent(page_number=2, text="Unit 2"),
        ],
        metadata={"page_errors": []},
    )
    assert document.to_dict() == {
        "document_id": "doc-1",
        "filename": "syllabus.pdf",
        "page_count": 2,
        "extraction_method": "pymupdf",
        "pages": [
            {"page_number": 1, "text": "Unit 1"},
            {"page_number": 2, "text": "Unit 2"},
        ],
        "metadata": {"page_errors": []},
    }


def test_page_boundaries_are_preserved() -> None:
    document = NormalizedDocument(
        document_id="d",
        filename="f.pdf",
        page_count=3,
        extraction_method="pymupdf",
        pages=[PageContent(page_number=n, text=f"page {n}") for n in (1, 2, 3)],
    )
    assert [p.page_number for p in document.pages] == [1, 2, 3]


def test_total_chars() -> None:
    document = NormalizedDocument(
        document_id="d",
        filename="f.pdf",
        page_count=2,
        extraction_method="ocr",
        pages=[PageContent(page_number=1, text="abc"), PageContent(page_number=2, text="defghi")],
    )
    assert document.total_chars() == 9