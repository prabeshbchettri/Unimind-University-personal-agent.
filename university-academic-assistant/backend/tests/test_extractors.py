"""Tests for the deterministic metadata/structure extractor."""

from tests.structure_factory import make_normalized

from app.structure import DeterministicMetadataExtractor, DocumentClassifier
from app.structure.extractors.base import ExtractionResult

extractor = DeterministicMetadataExtractor()


def extract(page_texts: list[str], doc_type: str | None = None) -> ExtractionResult:
    """Run the classifier (like the pipeline) then extract with that type."""
    document = make_normalized(page_texts)
    if doc_type is None:
        doc_type = DocumentClassifier().classify(document)
    return extractor.extract(document, doc_type)


def test_syllabus_metadata_and_topics() -> None:
    result = extract(
        [
            "TRIBHUVAN UNIVERSITY",
            "B.Sc. CSIT",
            "Semester V",
            "Database Management System Syllabus",
            "Unit 1: Introduction to Databases",
            "Unit 2: Normalization",
            "Unit 3: Transaction Processing",
            "Recommended Books: ...",
        ]
    )
    assert result.metadata.document_type == "syllabus"
    assert result.metadata.semester == 5
    assert result.metadata.subject == "Database Management System"
    assert result.metadata.university == "TRIBHUVAN UNIVERSITY"
    assert result.metadata.program == "B.Sc. CSIT"
    assert "Normalization" in result.structure.topics
    assert "Transaction Processing" in result.structure.topics


def test_syllabus_plain_topic_list() -> None:
    result = extract(
        [
            "Database Management System Syllabus",
            "Semester V",
            "Course Objectives",
            "Normalization",
            "Transaction Processing",
            "Indexing",
            "Recommended Books",
        ]
    )
    assert result.metadata.subject == "Database Management System"
    assert result.structure.topics == ["Normalization", "Transaction Processing", "Indexing"]


def test_syllabus_missing_metadata_not_fabricated() -> None:
    result = extract(["Database Management System Syllabus", "Unit 1: Normalization"])
    assert result.metadata.semester is None
    assert result.metadata.academic_year is None
    assert result.metadata.subject_code is None


def test_past_question_metadata_and_questions() -> None:
    result = extract(
        [
            "2080",
            "Database Management System",
            "Time: 3 Hrs.",
            "Full Marks: 60",
            "Q5. Explain normalization. 10 Marks",
            "Q2. Define DBMS. 5 Marks",
        ]
    )
    assert result.metadata.year == 2080
    assert result.metadata.subject == "Database Management System"
    assert result.metadata.question_number == 5
    assert result.metadata.marks == 10
    assert result.metadata.topic == "normalization"

    assert len(result.structure.questions) == 2
    first = result.structure.questions[0]
    assert first.question_number == 5
    assert first.marks == 10
    assert "normalization" in first.text.lower()


def test_past_question_missing_year() -> None:
    result = extract(["Database Management System", "Time: 3 Hrs.", "Q1. What is SQL? 5 Marks"])
    assert result.metadata.year is None
    assert result.metadata.question_number == 1


def test_notice_title_from_subject_line() -> None:
    result = extract(["NOTICE", "Subject: Mid Semester Examination", "Date: 2024-05-01", "To all students, exams begin Monday."])
    assert result.metadata.title == "Mid Semester Examination"


def test_rules_regulations_basic_metadata() -> None:
    result = extract(["Rules and Regulations", "Attendance requirements", "Eligibility criteria"])
    assert result.metadata.title == "Rules and Regulations"
    assert result.metadata.semester is None
    assert result.structure.chapters == []


def test_academic_calendar_metadata() -> None:
    result = extract(["Academic Calendar 2024/2025", "Classes commence on 2024-09-01."])
    assert result.metadata.title == "Academic Calendar 2024/2025"
    assert result.metadata.academic_year == "2024/2025"


def test_library_book_metadata_and_chapters() -> None:
    result = extract(
        [
            "Database Systems: Design, Implementation and Management",
            "by Carlos Coronel",
            "Contents",
            "Preface",
            "Chapter 1: Introduction",
            "1.1 Database Concepts",
            "1.2 Data Modeling",
            "Chapter 2: The Relational Model",
            "2.1 Tables",
            "ISBN 978-1-337-62790-0",
        ]
    )
    assert result.metadata.title == "Database Systems: Design, Implementation and Management"
    assert result.metadata.author == "Carlos Coronel"
    assert len(result.structure.chapters) == 2
    assert result.structure.chapters[0].title == "Introduction"
    assert "Database Concepts" in result.structure.chapters[0].subtopics


def test_library_book_remains_independent() -> None:
    """A library book must never be assigned a semester, subject or university."""
    result = extract(
        [
            "Operating System Concepts",
            "by Abraham Silberschatz",
            "Contents",
            "Chapter 1: Introduction",
            "1.1 What Operating Systems Do",
            "ISBN 978-1-119-45633-9",
        ]
    )
    assert result.metadata.semester is None
    assert result.metadata.subject is None
    assert result.metadata.university is None
    assert result.metadata.program is None


def test_unknown_document_has_null_metadata() -> None:
    result = extract(["Some random unrelated text about football matches and weather."])
    assert result.metadata.title is not None  # first substantial line is fine
    assert result.metadata.semester is None
    assert result.metadata.subject is None
    assert result.metadata.year is None
    assert result.metadata.university is None
    assert result.structure.questions == []
    assert result.structure.chapters == []