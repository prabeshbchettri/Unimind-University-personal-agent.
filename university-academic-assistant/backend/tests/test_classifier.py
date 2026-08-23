"""Tests for document type classification."""

from tests.structure_factory import make_normalized

from app.structure import DocumentClassifier

SYLLABUS = [
    "TRIBHUVAN UNIVERSITY",
    "Faculty of Science and Technology",
    "B.Sc. CSIT",
    "Semester V",
    "Database Management System Syllabus",
    "Unit 1: Introduction to Databases",
    "Unit 2: Normalization",
    "Unit 3: Transaction Processing",
    "Credit: 3, Teaching Hours: 45",
    "Recommended Books: ...",
]

PAST_QUESTION = [
    "2080",
    "Database Management System",
    "Time: 3 Hrs.",
    "Full Marks: 60",
    "Candidates are required to answer all questions.",
    "Q1. Define DBMS. 5 Marks",
    "Q5. Explain normalization. 10 Marks",
]

NOTICE = [
    "NOTICE",
    "Subject: Mid Semester Examination",
    "Date: 2024-05-01",
    "This is to inform all students that examinations will begin next week.",
    "All concerned are requested to follow the schedule.",
]

RULES = [
    "Tribhuvan University",
    "Rules and Regulations for Examinations",
    "Regulation 1: Admission",
    "Eligibility criteria for candidates.",
    "Attendance requirements.",
    "Code of Conduct and Disciplinary actions.",
    "Grievance procedures.",
]

CALENDAR = [
    "Academic Calendar 2024/2025",
    "Registration opens for new students.",
    "Classes commence on 2024-09-01.",
    "Mid-Semester Examination Schedule.",
    "Winter Break from December.",
    "Final examination deadlines.",
]

BOOK = [
    "Database Systems: Design, Implementation and Management",
    "by Carlos Coronel",
    "Contents",
    "Preface",
    "Chapter 1: Introduction",
    "1.1 Database Concepts",
    "1.2 Data Modeling",
    "Chapter 2: The Relational Model",
    "2.1 Tables",
    "Copyright (c) 2019",
    "Index",
    "ISBN 978-1-337-62790-0",
]

UNKNOWN = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello world, this content is unrelated to university documents.",
]


def classify(pages: list[str]) -> str:
    return DocumentClassifier().classify(make_normalized(pages))


def test_classifies_syllabus() -> None:
    assert classify(SYLLABUS) == "syllabus"


def test_classifies_past_question() -> None:
    assert classify(PAST_QUESTION) == "past_question"


def test_classifies_notice() -> None:
    assert classify(NOTICE) == "notice"


def test_classifies_rules_regulations() -> None:
    assert classify(RULES) == "rules_regulations"


def test_classifies_academic_calendar() -> None:
    assert classify(CALENDAR) == "academic_calendar"


def test_classifies_library_book() -> None:
    assert classify(BOOK) == "library_book"


def test_classifies_unknown() -> None:
    assert classify(UNKNOWN) == "unknown"


def test_filename_does_not_influence_classification() -> None:
    # A file named "syllabus.pdf" whose content is unrelated is still "unknown".
    document = make_normalized(UNKNOWN, filename="syllabus.pdf")
    assert DocumentClassifier().classify(document) == "unknown"


def test_empty_document_is_unknown() -> None:
    document = make_normalized(["", "  "])
    assert DocumentClassifier().classify(document) == "unknown"