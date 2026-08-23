"""Qdrant collection routing.

Documents are routed to one of three collections based on their type:

- ``university_docs`` — syllabi, notices, rules/regulations, calendar, unknown
- ``past_questions`` — past question papers
- ``library_books`` — library books

Qdrant is used for vector retrieval only; chat history (PostgreSQL) and graph
relationships (Neo4j) are stored elsewhere in later phases.
"""

from __future__ import annotations

from collections.abc import Sequence

DEFAULT_COLLECTIONS = ("university_docs", "past_questions", "library_books")

_UNIVERSITY_DOC_TYPES = {"syllabus", "notice", "rules_regulations", "academic_calendar", "unknown"}


def collection_for(document_type: str, collection_names: Sequence[str]) -> str:
    """Return the collection a document of ``document_type`` belongs to."""
    names = list(collection_names) or list(DEFAULT_COLLECTIONS)
    if document_type == "past_question":
        return names[1]
    if document_type == "library_book":
        return names[2]
    return names[0]


def all_collections(collection_names: Sequence[str]) -> list[str]:
    """Return every collection name (searchable in order)."""
    return list(collection_names) or list(DEFAULT_COLLECTIONS)
