"""Evaluation corpus (Phase 13).

A synthetic but representative academic corpus covering the seven categories
the project must handle:

1. Syllabus
2. Notices
3. Rules and regulations
4. Academic calendar
5. Past questions
6. Library books
7. External/current information (router-level only — never indexed)

Every document carries the metadata the real pipeline produces (document
type, title, subject, semester, author for books), and ``RETRIEVAL_QUERIES``
labels each query with its relevant documents so Recall@K / Precision@K /
MRR can be computed at the document level.
"""

from __future__ import annotations

from app.structure.models import (
    Chapter,
    DocumentMetadata,
    DocumentStructure,
    Question,
    StructuredDocument,
)
from app.ingestion.models import NormalizedDocument, PageContent

# ---------------------------------------------------------------------------
# Corpus documents
# ---------------------------------------------------------------------------

DBMS_SUBJECT = "Database Management System"

SYLLABUS_TOPICS = [
    ("Introduction", ["Data Models", "Database Architecture"]),
    ("Normalization", ["Normal Forms", "First Normal Form"]),
    ("Transactions", ["ACID Properties"]),
    ("Indexing", ["B-Trees"]),
]


def _structured(
    document_id: str,
    document_type: str,
    pages_text: list[str],
    *,
    title: str,
    subject: str | None = None,
    semester: int | None = None,
    topics: list[str] | None = None,
    subtopics: list[str] | None = None,
    chapters: list[tuple[str, list[str]]] | None = None,
    questions: list[Question] | None = None,
    author: str | None = None,
    year: int | None = None,
    academic_year: str | None = None,
) -> StructuredDocument:
    structure = DocumentStructure(
        topics=topics or [],
        subtopics=subtopics or [],
        chapters=[Chapter(chapter_number=i, title=t, topics=[t], subtopics=sub) for i, (t, sub) in enumerate(chapters or [], start=1)],
        questions=questions or [],
    )
    return StructuredDocument(
        document_id=document_id,
        filename=f"{document_id}.pdf",
        extraction_method="pymupdf",
        page_count=len(pages_text),
        pages=[PageContent(page_number=i, text=text) for i, text in enumerate(pages_text, start=1)],
        metadata=DocumentMetadata(
            document_type=document_type,
            title=title,
            subject=subject,
            semester=semester,
            author=author,
            year=year,
            academic_year=academic_year,
        ),
        structure=structure,
    )


def _syllabus_lines(subject: str, topics: list[tuple[str, list[str]]]) -> list[str]:
    lines = [f"{subject} Syllabus", ""]
    for index, (topic, subtopics) in enumerate(topics, start=1):
        lines.append(f"Unit {index}: {topic}")
        for sub in subtopics:
            lines.append(f"{index}.1 {sub}")
        lines.append(f"The unit covers the core ideas of {topic}.")
        lines.append("")
    return lines


def build_corpus() -> list[StructuredDocument]:
    """Return the evaluation corpus (deterministic, document ids stable)."""

    syllabus_text = "\n".join(_syllabus_lines(DBMS_SUBJECT, SYLLABUS_TOPICS))
    dbms_syllabus = _structured(
        "syllabus-dbms",
        "syllabus",
        [syllabus_text],
        title="DBMS Syllabus 2024",
        subject=DBMS_SUBJECT,
        semester=5,
        topics=[topic for topic, _ in SYLLABUS_TOPICS],
        chapters=SYLLABUS_TOPICS,
    )

    attendance_notice = _structured(
        "notice-attendance",
        "notice",
        [
            "Notice\n"
            "Attendance Requirement for 2026\n"
            "All students must attend at least 75 percent of classes in every "
            "subject. Students below the threshold are not eligible to sit the "
            "final examination. Attendance is recorded per lecture by the "
            "class teacher."
        ],
        title="Notice: Attendance Requirement 2026",
    )

    regulations = _structured(
        "regulations-exam",
        "rules_regulations",
        [
            "Examination Regulations 2024\n"
            "Rule 8: Grading\n"
            "Marks are converted to grade points: 80-100 is A, 60-79 is B, "
            "40-59 is C, below 40 is F. A student failing a subject must "
            "retake it in the next semester. Rule 12: the use of mobile "
            "phones inside the examination hall is prohibited."
        ],
        title="Examination Regulations 2024",
    )

    calendar = _structured(
        "calendar-2024",
        "academic_calendar",
        [
            "Academic Calendar 2024\n"
            "First semester: 2024-02-12 to 2024-06-28. Exam week starts on "
            "2024-06-17. Spring holidays: 2024-04-09 to 2024-04-15. "
            "Second semester begins 2024-07-15."
        ],
        title="Academic Calendar 2024",
    )

    past_paper = _structured(
        "pastpaper-dbms-2080",
        "past_question",
        [
            "DBMS Past Question Paper 2080\n"
            "Q1. Explain the concept of normalization and its normal forms. "
            "(10 marks)\n"
            "Q2. Write SQL queries for the student database. (8 marks)\n"
            "Q3. Discuss ACID properties of transactions. (7 marks)"
        ],
        title="DBMS Past Question Paper 2080",
        subject=DBMS_SUBJECT,
        semester=5,
        year=2080,
        topics=["Normalization", "SQL", "ACID Properties"],
        questions=[
            Question(question_number=1, text="Explain the concept of normalization and its normal forms.", marks=10),
            Question(question_number=2, text="Write SQL queries for the student database.", marks=8),
            Question(question_number=3, text="Discuss ACID properties of transactions.", marks=7),
        ],
    )

    dsa_syllabus = _structured(
        "syllabus-dsa",
        "syllabus",
        [
            "Data Structures and Algorithms Syllabus\n"
            "Unit 1: Arrays\n1.1 Array Operations\n"
            "Unit 2: Linked Lists\n2.1 Singly Linked Lists\n"
            "Unit 3: Trees\n3.1 Binary Trees\n"
            "Unit 4: Sorting\n4.1 Quick Sort\n"
        ],
        title="DSA Syllabus 2024",
        subject="Data Structures and Algorithms",
        semester=5,
        topics=["Arrays", "Linked Lists", "Trees", "Sorting"],
        chapters=[
            ("Arrays", ["Array Operations"]),
            ("Linked Lists", ["Singly Linked Lists"]),
            ("Trees", ["Binary Trees"]),
            ("Sorting", ["Quick Sort"]),
        ],
    )

    book_dbms = _structured(
        "book-dbms-concepts",
        "library_book",
        [
            "Chapter 1: Introduction\n1.1 Data Models\n\n"
            "Chapter 2: Normalization\n2.1 Normal Forms\n"
            "Normalization removes redundancy in tables.\n\n"
            "Chapter 3: Transactions\n3.1 ACID Properties\n"
            "Transactions keep data consistent.\n\n"
            "Chapter 4: Indexing\n4.1 B-Trees\nIndexes speed up queries."
        ],
        title="Database System Concepts",
        author="Abraham Silberschatz",
        topics=[topic for topic, _ in SYLLABUS_TOPICS],
        chapters=SYLLABUS_TOPICS,
    )

    book_os = _structured(
        "book-os-principles",
        "library_book",
        [
            "Chapter 1: Processes\n1.1 Scheduling\nProcess scheduling "
            "manages CPU time.\n\nChapter 2: Memory\n2.1 Paging\nMemory "
            "paging virtualizes physical memory."
        ],
        title="Operating System Principles",
        author="Abraham Silberschatz",
        topics=["Processes", "Memory"],
        chapters=[("Processes", ["Scheduling"]), ("Memory", ["Paging"])],
    )

    return [dbms_syllabus, dsa_syllabus, attendance_notice, regulations, calendar, past_paper, book_dbms, book_os]


DOCUMENTS = build_corpus()

# ---------------------------------------------------------------------------
# Labeled retrieval queries
# ---------------------------------------------------------------------------

RETRIEVAL_QUERIES: list[dict] = [
    # --- NORMAL: semantic / conceptual -------------------------------------
    {"category": "syllabus", "query": "Explain normalization.", "expected_docs": {"syllabus-dbms", "book-dbms-concepts"}, "expected_strategy": "NORMAL"},
    {"category": "syllabus", "query": "What are ACID properties?", "expected_docs": {"syllabus-dbms", "book-dbms-concepts"}, "expected_strategy": "NORMAL"},
    {"category": "syllabus", "query": "How do B-tree indexes speed up queries?", "expected_docs": {"syllabus-dbms", "book-dbms-concepts"}, "expected_strategy": "NORMAL"},
    # --- HYBRID: exact-token administrative facts ----------------------------
    {"category": "notice", "query": "What is the attendance requirement?", "expected_docs": {"notice-attendance"}, "expected_strategy": "HYBRID"},
    {"category": "rules_regulations", "query": "What does rule 8 say about grading?", "expected_docs": {"regulations-exam"}, "expected_strategy": "HYBRID"},
    {"category": "academic_calendar", "query": "When does exam week start in 2024?", "expected_docs": {"calendar-2024"}, "expected_strategy": "HYBRID"},
    # --- GRAPH: structural relationships --------------------------------------
    {"category": "syllabus", "query": "List the topics of Database Management System.", "expected_docs": {"syllabus-dbms"}, "expected_strategy": "GRAPH"},
    {"category": "past_questions", "query": "Which past questions are about normalization?", "expected_docs": {"pastpaper-dbms-2080"}, "expected_strategy": "GRAPH"},
    {"category": "syllabus", "query": "Which subjects are in semester 5?", "expected_docs": {"syllabus-dbms", "syllabus-dsa"}, "expected_strategy": "GRAPH"},
    # --- Library books -------------------------------------------------------
    {"category": "library_books", "query": "What does the Database System Concepts book cover?", "expected_docs": {"book-dbms-concepts"}, "expected_strategy": "NORMAL"},
    {"category": "library_books", "query": "Which book explains normal forms?", "expected_docs": {"book-dbms-concepts"}, "expected_strategy": "NORMAL"},
]


def retrieval_queries() -> list[dict]:
    return [dict(item) for item in RETRIEVAL_QUERIES]