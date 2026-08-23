"""Validation of LLM-produced structured output.

An LLM may return invalid, truncated or hallucinated JSON. Nothing reaches the
structured document models until it has been parsed and validated against the
Pydantic schema here. On any failure, :class:`LLMExtractionError` is raised and
the payload is discarded.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from app.structure.models import DocumentMetadata, DocumentType


class LLMExtractionError(Exception):
    """Raised when LLM output cannot be parsed or validated."""


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```", re.IGNORECASE)


def parse_llm_json(raw: str) -> dict:
    """Extract the first JSON object from an LLM response.

    Handles code fences and surrounding prose. Raises
    :class:`LLMExtractionError` when no valid JSON object can be found.
    """
    if not raw or not raw.strip():
        raise LLMExtractionError("LLM returned an empty response.")

    cleaned = _JSON_FENCE_RE.sub("", raw).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LLMExtractionError("No JSON object found in LLM response.")

    candidate = cleaned[start : end + 1]
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMExtractionError(f"LLM response is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise LLMExtractionError("LLM JSON payload is not an object.")
    return payload


def validate_llm_metadata(payload: dict, document_type: DocumentType) -> DocumentMetadata:
    """Validate a parsed LLM payload against the metadata schema."""
    payload = dict(payload)
    payload["document_type"] = document_type
    try:
        return DocumentMetadata.model_validate(payload)
    except ValidationError as exc:
        raise LLMExtractionError(f"LLM output failed schema validation: {exc}") from exc


def validate_structure_fields(payload: dict) -> dict[str, Any]:
    """Validate list-shaped structure fields, raising on malformed shapes.

    Accepts only proper lists of strings/objects; any other shape is treated as
    invalid LLM output.
    """
    for field in ("topics", "subtopics"):
        value = payload.get(field)
        if value is not None:
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise LLMExtractionError(f"LLM field {field!r} must be a list of strings.")

    questions = payload.get("questions")
    if questions is not None:
        if not isinstance(questions, list) or not all(isinstance(q, dict) for q in questions):
            raise LLMExtractionError("LLM field 'questions' must be a list of objects.")

    chapters = payload.get("chapters")
    if chapters is not None:
        if not isinstance(chapters, list) or not all(isinstance(c, dict) for c in chapters):
            raise LLMExtractionError("LLM field 'chapters' must be a list of objects.")

    return payload
