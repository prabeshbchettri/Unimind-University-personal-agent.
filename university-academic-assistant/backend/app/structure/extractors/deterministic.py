"""Deterministic metadata and structure extractor.

Uses the heuristics module. Only fields supported by the source content are
populated; everything else stays ``None`` / empty. Library books are never
assigned a university, semester or subject.
"""

from __future__ import annotations

from app.config import settings
from app.ingestion.models import NormalizedDocument
from app.structure import heuristics
from app.structure.models import (
    Chapter,
    DocumentMetadata,
    DocumentStructure,
    DocumentType,
    Question,
)

from .base import ExtractionResult, MetadataExtractor


def _extra_university_names() -> list[str]:
    raw = settings.university_names
    return [name.strip() for name in raw.split(",") if name.strip()]


class DeterministicMetadataExtractor(MetadataExtractor):
    """Rule-based extraction of academic metadata and structure."""

    name = "deterministic"

    def extract(
        self, document: NormalizedDocument, document_type: DocumentType
    ) -> ExtractionResult:
        text = "\n".join(page.text for page in document.pages)
        first_page = document.pages[0].text if document.pages else ""
        extra_universities = _extra_university_names()

        metadata = DocumentMetadata(document_type=document_type)
        structure = DocumentStructure()

        metadata.university = heuristics.find_university(text, extra_universities)
        metadata.program = heuristics.find_program(text)
        metadata.semester = heuristics.find_semester(text)
        metadata.academic_year = heuristics.find_academic_year(text)
        metadata.subject_code = heuristics.find_subject_code(text)
        metadata.author = heuristics.find_author(text)

        if document_type == "syllabus":
            metadata.title = first_substantial_title(first_page)
            metadata.subject = heuristics.subject_from_syllabus(text) or metadata.title
            topics = heuristics.extract_syllabus_topics(text)
            structure.topics = topics
            metadata.topic = topics[0] if topics else None

        elif document_type == "past_question":
            metadata.title = first_substantial_title(first_page)
            metadata.year = heuristics.find_year(text)
            metadata.subject = heuristics.subject_from_past_paper(text)
            questions = heuristics.extract_questions(text)
            structure.questions = [Question(**question) for question in questions]
            if questions:
                metadata.question_number = questions[0]["question_number"]
                metadata.marks = questions[0]["marks"]
                metadata.topic = heuristics.topic_from_question(questions[0]["text"])

        elif document_type == "library_book":
            metadata.title = first_substantial_title(first_page)
            metadata.author = heuristics.find_author(text) or metadata.author
            chapters = heuristics.extract_chapters(text)
            subtopic_map = heuristics.extract_subtopic_map(text)
            structure.chapters = [
                Chapter(
                    chapter_number=chapter["chapter_number"],
                    title=chapter["title"],
                    topics=[chapter["title"]] if chapter["title"] else [],
                    subtopics=[
                        title
                        for key, title in subtopic_map.items()
                        if str(chapter["chapter_number"] or "") in key.split(".")[0:1]
                    ],
                )
                for chapter in chapters
            ]
            structure.topics = [chapter["title"] for chapter in chapters if chapter["title"]]

        elif document_type == "notice":
            metadata.title = heuristics.subject_line(text) or first_substantial_title(first_page)

        else:  # rules_regulations, academic_calendar, unknown
            metadata.title = first_substantial_title(first_page)

        return ExtractionResult(metadata=metadata, structure=structure)


def first_substantial_title(first_page_text: str) -> str | None:
    """Title heuristic: the first substantial line of the first page."""
    return heuristics.first_substantial_line(first_page_text)
