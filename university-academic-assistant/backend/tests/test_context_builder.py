"""Tests for the context builder (Phase 5)."""

from tests import rag_factory as factory

from app.rag.context import NO_CONTEXT_NOTE, SYSTEM_PROMPT


def test_system_prompt_encodes_grounding_rules() -> None:
    assert "Answer using the supplied retrieved context" in SYSTEM_PROMPT
    assert "Do not invent university-specific facts" in SYSTEM_PROMPT
    assert "insufficient" in SYSTEM_PROMPT
    assert "Prefer retrieved university information over general knowledge" in SYSTEM_PROMPT
    assert "source and page" in SYSTEM_PROMPT


def test_build_separates_context_and_query() -> None:
    builder = factory.make_context_builder()
    system, prompt = builder.build("Explain normalization.", [factory.normalization_chunk()])

    assert system == SYSTEM_PROMPT
    assert "CONTEXT:" in prompt
    assert "--- END OF CONTEXT ---" in prompt
    assert "User Query: Explain normalization." in prompt
    # Query text must not appear inside the CONTEXT section.
    context_section = prompt.split("--- END OF CONTEXT ---")[0]
    assert "Explain normalization." not in context_section


def test_build_includes_source_metadata_and_text() -> None:
    builder = factory.make_context_builder()
    _, prompt = builder.build("Explain normalization.", [factory.normalization_chunk()])

    assert "[Source 1]" in prompt
    assert "title: DBMS Syllabus" in prompt
    assert "page: 3" in prompt
    assert "subject: Database Management System" in prompt
    assert "topic: Normalization" in prompt
    assert "Normalization removes redundancy in tables." in prompt


def test_build_uses_numbered_sources() -> None:
    builder = factory.make_context_builder()
    chunks = [
        factory.normalization_chunk(),
        factory.SearchResult(text="Databases store data.", score=0.8, metadata={"title": "DBMS Syllabus"}, collection="university_docs"),
    ]
    _, prompt = builder.build("databases", chunks)

    assert "[Source 1]" in prompt
    assert "[Source 2]" in prompt


def test_build_empty_context_notes_insufficiency() -> None:
    builder = factory.make_context_builder()
    _, prompt = builder.build("When is the exam?", [])

    assert NO_CONTEXT_NOTE in prompt


def test_build_respects_max_chars() -> None:
    builder = factory.make_context_builder(max_chars=120)
    big_chunk = factory.normalization_chunk()
    big_chunk.text = "word " * 200
    _, prompt = builder.build("normalization", [big_chunk])

    context_section = prompt.split("--- END OF CONTEXT ---")[0]
    assert len(context_section) <= 130


def test_build_drops_sources_beyond_limit() -> None:
    builder = factory.make_context_builder(max_chars=200, max_sources=8)
    chunks = [
        factory.SearchResult(text="word " * 100, score=1.0, metadata={"title": "A"}, collection="university_docs"),
        factory.SearchResult(text="more context here", score=0.5, metadata={"title": "B"}, collection="university_docs"),
    ]
    _, prompt = builder.build("query", chunks)

    assert "[Source 1]" in prompt
    assert "[Source 2]" not in prompt