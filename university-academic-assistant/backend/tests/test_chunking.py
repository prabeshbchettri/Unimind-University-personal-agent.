"""Tests for semantic chunking."""

from tests.structure_factory import make_structured

from app.chunking import SemanticChunker, split_paragraphs
from app.structure.models import DocumentMetadata, DocumentStructure


def syllabus_doc():
    return make_structured(
        [
            "Unit 1: Introduction to Databases\n"
            "Databases store data. A DBMS is software that manages that data.\n"
            "\n"
            "Unit 2: Normalization\n"
            "Normalization removes redundancy in relational tables.",
        ],
        document_type="syllabus",
        metadata=DocumentMetadata(
            document_type="syllabus",
            title="DBMS Syllabus",
            semester=5,
            subject="Database Management System",
            subject_code="CSC-301",
        ),
        structure=DocumentStructure(topics=["Introduction to Databases", "Normalization"]),
    )


def test_syllabus_topics_stay_together() -> None:
    chunks = SemanticChunker(max_chars=140, overlap_chars=0).chunk(syllabus_doc())

    assert len(chunks) >= 2
    normalization = next(c for c in chunks if c.topic == "Normalization")
    # Heading, definition and explanation are kept together in one chunk.
    assert "Unit 2: Normalization" in normalization.text
    assert "Normalization removes redundancy in relational tables." in normalization.text
    assert normalization.section == "topic"
    assert normalization.page == 1
    assert normalization.document_id == "doc-1"


def test_chunk_metadata_preserved() -> None:
    chunk = SemanticChunker(max_chars=140, overlap_chars=0).chunk(syllabus_doc())[0]
    assert chunk.semester == 5
    assert chunk.subject == "Database Management System"
    assert chunk.subject_code == "CSC-301"
    assert chunk.title == "DBMS Syllabus"

    payload = chunk.payload()
    assert payload["document_id"] == "doc-1"
    assert payload["document_type"] == "syllabus"
    assert payload["subject"] == "Database Management System"
    assert payload["text"]


def test_chunk_overlap_carries_tail_context() -> None:
    structured = make_structured(
        [
            "Unit 1: A\nsome text about A\n"
            "Unit 2: B\nsome text about B\n"
            "Unit 3: C\nsome text about C",
        ],
        document_type="syllabus",
        structure=DocumentStructure(topics=["A", "B", "C"]),
    )
    chunks = SemanticChunker(max_chars=50, overlap_chars=30).chunk(structured)

    assert len(chunks) >= 2
    # The tail block of the previous chunk (B) is carried into the next chunk,
    # so B appears both where it belongs and in the overlapping successor.
    assert "some text about B" in chunks[1].text
    assert "some text about B" in chunks[2].text


def test_past_question_per_question_chunks() -> None:
    structured = make_structured(
        [
            "2080\n"
            "Database Management System\n"
            "Time: 3 Hrs.\n"
            "Full Marks: 60\n"
            "Q1. Define DBMS. 5 Marks\n"
            "DBMS manages data.\n"
            "Q5. Explain normalization. 10 Marks\n"
            "Normalization removes redundancy.",
        ],
        document_type="past_question",
        metadata=DocumentMetadata(document_type="past_question", subject="Database Management System", year=2080),
    )
    chunks = SemanticChunker(max_chars=80, overlap_chars=0).chunk(structured)

    q5 = next(c for c in chunks if c.question_number == 5)
    assert q5.marks == 10
    assert q5.section == "question"
    assert q5.topic == "normalization"
    assert "Normalization removes redundancy." in q5.text

    q1 = next(c for c in chunks if c.question_number == 1)
    assert q1.marks == 5


def test_library_book_chapters_and_subtopics() -> None:
    structured = make_structured(
        [
            "Database Systems\n"
            "by Carlos Coronel\n"
            "Contents\n"
            "Chapter 1: Introduction\n"
            "1.1 Database Concepts\n"
            "Concepts explained here.\n"
            "1.2 Data Modeling\n"
            "Modeling explained.\n"
            "Chapter 2: Relational Model\n"
            "2.1 Tables",
        ],
        document_type="library_book",
        metadata=DocumentMetadata(document_type="library_book", title="Database Systems", author="Carlos Coronel"),
        structure=DocumentStructure(chapters=[]),
    )
    chunks = SemanticChunker(max_chars=50, overlap_chars=0).chunk(structured)

    concept = next(c for c in chunks if c.subtopic == "Database Concepts")
    assert concept.topic == "Introduction"
    assert concept.section == "chapter"
    assert "Concepts explained here." in concept.text

    chapter2 = next(c for c in chunks if c.topic == "Relational Model")
    assert "2.1 Tables" in chapter2.text


def test_generic_paragraph_chunking() -> None:
    structured = make_structured(
        ["NOTICE\n\nSubject: Mid Semester Examination\n\nAll students must appear for the exam."],
        document_type="notice",
        metadata=DocumentMetadata(document_type="notice", title="Notice"),
    )
    chunks = SemanticChunker().chunk(structured)

    assert len(chunks) == 1
    assert chunks[0].page == 1
    assert "mid semester examination" in chunks[0].text.lower()


def test_oversized_block_split_at_sentence_boundaries() -> None:
    paragraph = (
        "This is the first sentence about database systems. "
        "This is the second sentence about normalization. "
        "This is the third sentence about transaction processing. "
        "This is the fourth sentence about indexing and queries. "
        "This is the fifth sentence about concurrency control. "
        "This is the sixth sentence about recovery mechanisms."
    )
    structured = make_structured([paragraph], document_type="unknown")
    chunks = SemanticChunker(max_chars=120, overlap_chars=0).chunk(structured)

    assert len(chunks) >= 2
    # No chunk may cut a sentence in half.
    for chunk in chunks:
        assert chunk.text.rstrip().endswith((".", "!", "?"))


def test_split_paragraphs() -> None:
    assert split_paragraphs("a\n\nb\nc\n\nd") == ["a", "b\nc", "d"]
    assert split_paragraphs("") == []


def test_empty_document_yields_no_chunks() -> None:
    structured = make_structured([], document_type="syllabus")
    assert SemanticChunker().chunk(structured) == []


def test_deterministic_reproducibility_of_chunk_ids() -> None:
    a = SemanticChunker().chunk(syllabus_doc())
    b = SemanticChunker().chunk(syllabus_doc())
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_syllabus_units_become_distinct_chunks() -> None:
    chunks = SemanticChunker(max_chars=50, overlap_chars=0).chunk(
        make_structured(
            [
                "Unit 1: A\nsome text about A\n"
                "Unit 2: B\nsome text about B\n"
                "Unit 3: C\nsome text about C",
            ],
            document_type="syllabus",
            structure=DocumentStructure(topics=["A", "B", "C"]),
        )
    )
    assert len(chunks) == 3
    assert {c.topic for c in chunks} == {"A", "B", "C"}