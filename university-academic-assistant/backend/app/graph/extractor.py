"""Deterministic entity + relationship extraction (Phase 7).

Converts a :class:`StructuredDocument` into graph nodes/relationships using
the metadata and structure produced by the Phase 3 pipeline. No LLM is
involved: every entity and relationship is derived from validated fields, so
nothing unsupported can be fabricated. (If an LLM extractor is added later,
``app.graph.schema.validate_relationships`` must gate its output.)

Extraction is intentionally focused: only important academic entities
(university, program, semester, subject, topics/subtopics, past questions) are
represented — not chunks or sentences.
"""

from __future__ import annotations

from typing import Any

from app.graph.models import (
    GraphExtraction,
    GraphNode,
    GraphRelationship,
)
from app.graph.schema import deduplicate_nodes, deduplicate_relationships, node_key
from app.structure.models import Question, StructuredDocument

# Document types whose structure.topics become Subject -> HAS_TOPIC -> Topic.
_HAS_TOPIC_TYPES = {"syllabus"}
# Document types that contribute Topic/Subtopic nodes at all.
_TOPIC_TYPES = {"syllabus", "past_question"}


def _page_of(structured: StructuredDocument, needle: str) -> int:
    """First page (1-based) containing ``needle``, else 1."""
    lowered = needle.lower()
    for page in structured.pages:
        if lowered in page.text.lower():
            return page.page_number
    return 1


class GraphExtractor:
    """Extracts a focused academic graph from a structured document."""

    name = "deterministic"

    def extract(self, structured: StructuredDocument) -> GraphExtraction:
        metadata = structured.metadata
        structure = structured.structure

        nodes: list[GraphNode] = []
        relationships: list[GraphRelationship] = []

        def source(page: int | None = None) -> dict[str, Any]:
            return {
                "document_id": structured.document_id,
                "page": page or 1,
                "source_type": metadata.document_type,
                "title": metadata.title or structured.filename,
            }

        # --- University / Program ------------------------------------------
        university_key: str | None = None
        program_key: str | None = None
        if metadata.university:
            university_key = node_key("University", metadata.university)
            nodes.append(GraphNode(label="University", key=university_key, props={"name": metadata.university, **source()}))
        if metadata.program:
            program_key = node_key("Program", metadata.program)
            nodes.append(GraphNode(label="Program", key=program_key, props={"name": metadata.program, **source()}))
            if university_key:
                relationships.append(
                    GraphRelationship(
                        type="HAS_PROGRAM",
                        start_label="University",
                        start_key=university_key,
                        end_label="Program",
                        end_key=program_key,
                        props=source(),
                    )
                )

        # --- Semester -------------------------------------------------------
        semester_key: str | None = None
        if metadata.semester is not None:
            semester_key = node_key("Semester", metadata.semester)
            nodes.append(GraphNode(label="Semester", key=semester_key, props={"semester": metadata.semester, **source()}))
            if program_key:
                relationships.append(
                    GraphRelationship(
                        type="HAS_SEMESTER",
                        start_label="Program",
                        start_key=program_key,
                        end_label="Semester",
                        end_key=semester_key,
                        props=source(),
                    )
                )

        # --- Subject --------------------------------------------------------
        subject_key: str | None = None
        if metadata.subject:
            subject_key = node_key("Subject", metadata.subject)
            nodes.append(
                GraphNode(
                    label="Subject",
                    key=subject_key,
                    props={
                        "name": metadata.subject,
                        "code": metadata.subject_code,
                        "semester": metadata.semester,
                        "program": metadata.program,
                        **source(),
                    },
                )
            )
            if semester_key:
                relationships.append(
                    GraphRelationship(
                        type="HAS_SUBJECT",
                        start_label="Semester",
                        start_key=semester_key,
                        end_label="Subject",
                        end_key=subject_key,
                        props=source(),
                    )
                )

        # --- Topics ---------------------------------------------------------
        # Topic nodes come from syllabus topics (linked to the subject) and
        # past-paper topics (linked only via PastQuestion -> ABOUT, so a paper
        # does not pollute the subject's topic list).
        topic_keys: dict[str, str] = {}
        if metadata.document_type in _TOPIC_TYPES:
            for topic in structure.topics:
                topic = topic.strip()
                if not topic:
                    continue
                topic_key = node_key("Topic", topic)
                topic_keys[topic_key] = topic
                nodes.append(GraphNode(label="Topic", key=topic_key, props={"name": topic, **source(_page_of(structured, topic))}))
                if subject_key and metadata.document_type in _HAS_TOPIC_TYPES:
                    relationships.append(
                        GraphRelationship(
                            type="HAS_TOPIC",
                            start_label="Subject",
                            start_key=subject_key,
                            end_label="Topic",
                            end_key=topic_key,
                            props=source(),
                        )
                    )

        # --- Subtopics ------------------------------------------------------
        # Deterministic mapping: a single explicit (topic, subtopic) pair from
        # metadata, or all subtopics under the only topic in the structure.
        if metadata.document_type in _TOPIC_TYPES:
            explicit_pair = bool(metadata.topic and metadata.subtopic)
            if explicit_pair:
                topic_key = node_key("Topic", metadata.topic)
                subtopic_key = node_key("Subtopic", metadata.subtopic)
                nodes.append(GraphNode(label="Subtopic", key=subtopic_key, props={"name": metadata.subtopic, **source(_page_of(structured, metadata.subtopic))}))
                relationships.append(
                    GraphRelationship(
                        type="HAS_SUBTOPIC",
                        start_label="Topic",
                        start_key=topic_key,
                        end_label="Subtopic",
                        end_key=subtopic_key,
                        props=source(),
                    )
                )
            elif len(structure.topics) == 1 and structure.subtopics:
                topic_key = node_key("Topic", structure.topics[0])
                for subtopic in structure.subtopics:
                    subtopic = subtopic.strip()
                    if not subtopic:
                        continue
                    subtopic_key = node_key("Subtopic", subtopic)
                    nodes.append(GraphNode(label="Subtopic", key=subtopic_key, props={"name": subtopic, **source(_page_of(structured, subtopic))}))
                    relationships.append(
                        GraphRelationship(
                            type="HAS_SUBTOPIC",
                            start_label="Topic",
                            start_key=topic_key,
                            end_label="Subtopic",
                            end_key=subtopic_key,
                            props=source(),
                        )
                    )

        # --- Past questions -------------------------------------------------
        if metadata.document_type == "past_question":
            questions: list[Question] = list(structure.questions)
            if not questions and metadata.question_number is not None:
                questions = [Question(question_number=metadata.question_number, text="", marks=metadata.marks)]
            for question in questions:
                qnum = question.question_number
                question_key = node_key("PastQuestion", (structured.document_id, qnum))
                page = _page_of(structured, question.text or str(qnum))
                nodes.append(
                    GraphNode(
                        label="PastQuestion",
                        key=question_key,
                        props={
                            "question_number": qnum,
                            "text": question.text,
                            "marks": question.marks,
                            "year": metadata.year if metadata.year is not None else metadata.academic_year,
                            "subject": metadata.subject,
                            **source(page),
                        },
                    )
                )
                if subject_key:
                    relationships.append(
                        GraphRelationship(
                            type="HAS_QUESTION",
                            start_label="Subject",
                            start_key=subject_key,
                            end_label="PastQuestion",
                            end_key=question_key,
                            props=source(page),
                        )
                    )
                # ABOUT topic: use the document topic when known, otherwise the
                # first structure topic literally named in the question text.
                about_topic = metadata.topic
                if not about_topic and question.text:
                    for topic in structure.topics:
                        if topic.lower() in question.text.lower():
                            about_topic = topic
                            break
                if about_topic:
                    relationships.append(
                        GraphRelationship(
                            type="ABOUT",
                            start_label="PastQuestion",
                            start_key=question_key,
                            end_label="Topic",
                            end_key=node_key("Topic", about_topic),
                            props=source(page),
                        )
                    )

        # Collapse duplicate entities/relationships within this document.
        return GraphExtraction(
            nodes=deduplicate_nodes(nodes),
            relationships=deduplicate_relationships(relationships),
        )
