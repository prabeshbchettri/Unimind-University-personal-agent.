"""Tests for the structure pipeline (classification + extraction + validation)."""

from pathlib import Path

from tests.structure_factory import make_normalized

from app.structure import StructurePipeline


def test_pipeline_produces_validated_structured_document() -> None:
    document = make_normalized(
        [
            "TRIBHUVAN UNIVERSITY",
            "B.Sc. CSIT",
            "Semester V",
            "Database Management System Syllabus",
            "Unit 1: Introduction",
            "Unit 2: Normalization",
        ],
        filename="dbms_syllabus.pdf",
        document_id="syllabus-1",
    )
    structured = StructurePipeline().process(document)

    assert structured.document_id == "syllabus-1"
    assert structured.filename == "dbms_syllabus.pdf"
    assert structured.metadata.document_type == "syllabus"
    assert structured.metadata.semester == 5
    assert structured.pages[0].text.startswith("TRIBHUVAN")
    # Schema-validated: metadata_payload is a plain dict with document fields.
    payload = structured.metadata_payload()
    assert payload["document_type"] == "syllabus"
    assert payload["semester"] == 5


def test_pipeline_past_question_end_to_end() -> None:
    document = make_normalized(
        ["2080", "Database Management System", "Q5. Explain normalization. 10 Marks"]
    )
    structured = StructurePipeline().process(document)

    assert structured.metadata.document_type == "past_question"
    assert structured.metadata.year == 2080
    assert structured.metadata.subject == "Database Management System"
    assert structured.metadata.question_number == 5
    assert structured.metadata.marks == 10
    assert structured.structure.questions[0].marks == 10


def test_pipeline_library_book_keeps_independence() -> None:
    document = make_normalized(
        [
            "Operating System Concepts",
            "by Abraham Silberschatz",
            "Contents",
            "Chapter 1: Introduction",
            "1.1 What Operating Systems Do",
        ]
    )
    structured = StructurePipeline().process(document)

    assert structured.metadata.document_type == "library_book"
    assert structured.metadata.author == "Abraham Silberschatz"
    assert structured.metadata.semester is None
    assert structured.metadata.subject is None
    assert structured.structure.chapters[0].title == "Introduction"


def test_pipeline_unknown_document() -> None:
    document = make_normalized(
        ["Completely unrelated content about sports and weather conditions."]
    )
    structured = StructurePipeline().process(document)
    assert structured.metadata.document_type == "unknown"


def test_pipeline_from_real_pdf(text_pdf: Path) -> None:
    """End-to-end: ingest a synthetic PDF, then structure it."""
    from app.ingestion import IngestionPipeline

    ingested = IngestionPipeline().ingest_file(text_pdf)
    structured = StructurePipeline().process(ingested)

    assert structured.metadata.document_type == "syllabus"
    assert structured.metadata.subject == "Database Management Systems"
    assert len(structured.pages) == 2