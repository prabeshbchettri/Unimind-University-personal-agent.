"""LLM-assisted metadata and structure extractor.

Wraps an LLM ``generate`` callable behind the :class:`MetadataExtractor`
interface. All LLM output is parsed and schema-validated before it is
returned; invalid or hallucinated JSON raises :class:`LLMExtractionError` and
never reaches the database layer.

This extractor is opt-in (``STRUCTURE_EXTRACTOR=llm``) and requires an LLM
service, which becomes available from Phase 5 onward. Until then the default
:class:`DeterministicMetadataExtractor` is used.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from app.ingestion.models import NormalizedDocument
from app.structure.models import (
    Chapter,
    DocumentMetadata,
    DocumentStructure,
    DocumentType,
    Question,
)
from app.structure.validation import (
    LLMExtractionError,
    parse_llm_json,
    validate_llm_metadata,
    validate_structure_fields,
)

from .base import ExtractionResult, MetadataExtractor

_PROMPT_TEMPLATE = """You are a document metadata extractor for a university academic assistant.

Classify and extract metadata + structure from the extracted text of a
{document_type} document.

Return ONLY a JSON object with these optional keys:
title, university, program, semester (int), subject, subject_code,
academic_year, year (int), author,
topics (array of strings), subtopics (array of strings),
questions (array of {{question_number (int), text (str), marks (int)}}),
chapters (array of {{chapter_number (int), title (str), topics (array), subtopics (array)}}).

Do NOT invent values. Use null for unknown fields. Library books must NOT be
assigned a university, semester, subject or program.

Extracted text:
<text>
{text}
</text>
"""


def _build_prompt(document: NormalizedDocument, document_type: DocumentType) -> str:
    text = "\n\n".join(page.text for page in document.pages)
    truncated = text[:12000]
    return _PROMPT_TEMPLATE.format(document_type=document_type, text=truncated)


class LLMMetadataExtractor(MetadataExtractor):
    """Extract metadata/structure from an LLM, with strict validation."""

    name = "llm"

    def __init__(self, generate: Callable[[str], str]) -> None:
        self.generate = generate

    def extract(
        self, document: NormalizedDocument, document_type: DocumentType
    ) -> ExtractionResult:
        prompt = _build_prompt(document, document_type)
        raw = self.generate(prompt)
        payload = parse_llm_json(raw)
        validate_structure_fields(payload)

        metadata = validate_llm_metadata(payload, document_type)
        structure = _payload_to_structure(payload)
        return ExtractionResult(metadata=metadata, structure=structure)


def _payload_to_structure(payload: dict[str, Any]) -> DocumentStructure:
    return DocumentStructure(
        topics=payload.get("topics") or [],
        subtopics=payload.get("subtopics") or [],
        questions=[
            Question(**question)
            for question in payload.get("questions") or []
        ],
        chapters=[
            Chapter(
                chapter_number=chapter.get("chapter_number"),
                title=str(chapter.get("title", "")),
                topics=chapter.get("topics") or [],
                subtopics=chapter.get("subtopics") or [],
            )
            for chapter in payload.get("chapters") or []
        ],
    )


def json_schema_prompt() -> str:
    """Return the JSON schema an LLM is asked to conform to (for reference)."""
    return json.dumps(
        {
            "type": "object",
            "properties": {
                "title": {"type": ["string", "null"]},
                "university": {"type": ["string", "null"]},
                "program": {"type": ["string", "null"]},
                "semester": {"type": ["integer", "null"]},
                "subject": {"type": ["string", "null"]},
                "subject_code": {"type": ["string", "null"]},
                "academic_year": {"type": ["string", "null"]},
                "year": {"type": ["integer", "null"]},
                "author": {"type": ["string", "null"]},
                "topics": {"type": "array", "items": {"type": "string"}},
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question_number": {"type": ["integer", "null"]},
                            "text": {"type": "string"},
                            "marks": {"type": ["integer", "null"]},
                        },
                    },
                },
            },
        }
    )


__all__ = [
    "LLMExtractionError",
    "LLMMetadataExtractor",
    "build_prompt",
    "json_schema_prompt",
]

# Aliased for backwards-friendly imports below.
build_prompt = _build_prompt
