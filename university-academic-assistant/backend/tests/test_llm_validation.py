"""Tests for LLM output parsing/validation and the LLM extractor.

No real LLM is required: a fake ``generate`` callable is injected.
"""

import pytest

from app.structure import LLMExtractionError, parse_llm_json
from app.structure.extractors.llm import LLMMetadataExtractor
from app.structure.validation import validate_llm_metadata, validate_structure_fields
from tests.structure_factory import make_normalized


def test_parse_plain_json() -> None:
    assert parse_llm_json('{"semester": 5, "subject": "DBMS"}') == {
        "semester": 5,
        "subject": "DBMS",
    }


def test_parse_fenced_json() -> None:
    raw = "Here is the result:\n```json\n{\"title\": \"Syllabus\"}\n```\nDone."
    assert parse_llm_json(raw) == {"title": "Syllabus"}


def test_parse_json_with_prose() -> None:
    raw = 'Sure! {"year": 2080} hope that helps.'
    assert parse_llm_json(raw) == {"year": 2080}


def test_parse_invalid_json_raises() -> None:
    with pytest.raises(LLMExtractionError):
        parse_llm_json('{"semester": 5, "subject": }')


def test_parse_no_json_raises() -> None:
    with pytest.raises(LLMExtractionError):
        parse_llm_json("I could not extract anything.")


def test_parse_empty_raises() -> None:
    with pytest.raises(LLMExtractionError):
        parse_llm_json("")


def test_validate_metadata_valid() -> None:
    metadata = validate_llm_metadata(
        {"title": "DBMS", "semester": 5, "subject": "Database Management System"},
        "syllabus",
    )
    assert metadata.document_type == "syllabus"
    assert metadata.semester == 5


def test_validate_metadata_wrong_type_raises() -> None:
    with pytest.raises(LLMExtractionError):
        validate_llm_metadata({"semester": "five"}, "syllabus")


def test_validate_structure_fields_malformed_raises() -> None:
    with pytest.raises(LLMExtractionError):
        validate_structure_fields({"topics": "not a list"})
    with pytest.raises(LLMExtractionError):
        validate_structure_fields({"questions": [1, 2]})


def test_llm_extractor_invalid_output_is_rejected() -> None:
    def bad_generate(prompt: str) -> str:
        return '{"semester": "not-an-int"}'

    extractor = LLMMetadataExtractor(generate=bad_generate)
    document = make_normalized(["some text"])
    with pytest.raises(LLMExtractionError):
        extractor.extract(document, "syllabus")


def test_llm_extractor_hallucinated_shape_is_rejected() -> None:
    def bad_generate(prompt: str) -> str:
        return '{"title": "X", "topics": [1, 2, 3]}'

    extractor = LLMMetadataExtractor(generate=bad_generate)
    with pytest.raises(LLMExtractionError):
        extractor.extract(make_normalized(["text"]), "syllabus")


def test_llm_extractor_valid_output() -> None:
    def good_generate(prompt: str) -> str:
        return (
            '{"title": "DBMS Syllabus", "semester": 5, "subject": "DBMS",'
            ' "topics": ["Normalization", "Indexing"]}'
        )

    extractor = LLMMetadataExtractor(generate=good_generate)
    result = extractor.extract(make_normalized(["text"]), "syllabus")
    assert result.metadata.semester == 5
    assert result.structure.topics == ["Normalization", "Indexing"]