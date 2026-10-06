"""Tests for context construction (selection, dedupe, budget, citations)."""

from __future__ import annotations

import pytest

from app.context import build_context, render_context, truncate_for_prompt
from app.retriever import RetrievedChunk
from tests.conftest import make_retrieved


def test_build_context_keeps_relevant_chunks_with_metadata() -> None:
    chunks = [
        make_retrieved("attendance requires seventy five percent", score=0.8),
        make_retrieved("exams need good standing", document="exams.pdf", page=2, score=0.6),
    ]
    context = build_context(chunks, min_score=0.5, max_chars=1000)

    assert context.enough_evidence is True
    assert [b.document for b in context.blocks] == ["doc.pdf", "exams.pdf"]
    assert [b.page for b in context.blocks] == [1, 2]
    assert [b.score for b in context.blocks] == [0.8, 0.6]


def test_build_context_drops_below_threshold_chunks() -> None:
    chunks = [
        make_retrieved("relevant one", score=0.9),
        make_retrieved("weak", score=0.3),
        make_retrieved("weaker", score=0.1),
    ]
    context = build_context(chunks, min_score=0.5, max_chars=1000)

    assert [b.text for b in context.blocks] == ["relevant one"]


def test_build_context_removes_near_duplicates() -> None:
    base = "Students must attend at least 75 percent of all scheduled sessions."
    chunks = [
        make_retrieved(base, score=0.9),
        # overlapping chunk repeating the same passage plus a tail
        make_retrieved(base + " Additional repeated content here.", score=0.85),
        make_retrieved("Library loans allow six books.", score=0.7),
    ]
    context = build_context(chunks, min_score=0.5, max_chars=5000)

    assert [b.text for b in context.blocks] == [base, "Library loans allow six books."]


def test_build_context_respects_character_budget() -> None:
    chunks = [
        make_retrieved("a" * 100, score=0.9),
        make_retrieved("b" * 100, score=0.8),
        make_retrieved("c" * 40, score=0.7),
    ]
    context = build_context(chunks, min_score=0.5, max_chars=150)

    # 'a' fits; 'b' (100+100 > 150) is skipped; 'c' still fits after 'a'.
    assert [b.text for b in context.blocks] == ["a" * 100, "c" * 40]
    assert context.total_chars == 140
    assert context.total_chars <= 150


def test_build_context_empty_result_is_insufficient_evidence() -> None:
    chunks = [make_retrieved("weak", score=0.2)]
    context = build_context(chunks, min_score=0.5, max_chars=1000)

    assert context.enough_evidence is False
    assert context.blocks == []
    assert render_context(context) == ""


def test_render_context_numbers_blocks_and_keeps_citation_metadata() -> None:
    chunks = [
        make_retrieved("first evidence", document="a.pdf", page=3, score=0.9),
        make_retrieved("second evidence", document="b.pdf", page=7, score=0.8),
    ]
    rendered = render_context(build_context(chunks, min_score=0.5, max_chars=1000))

    assert "[1] source: a.pdf, page 3" in rendered
    assert "[2] source: b.pdf, page 7" in rendered
    assert "first evidence" in rendered and "second evidence" in rendered


def test_truncate_for_prompt_caps_length() -> None:
    assert truncate_for_prompt("x" * 50, 10) == "x" * 10 + " …"
    assert truncate_for_prompt("short", 10) == "short"
    with pytest.raises(ValueError):
        truncate_for_prompt("x", 0)


def test_build_context_rejects_non_positive_budget() -> None:
    with pytest.raises(ValueError):
        build_context([make_retrieved("t")], min_score=0.5, max_chars=0)


def _lexical_chunk(chunk_id: str, text: str, bm25_score: float, page: int = 1) -> RetrievedChunk:
    """A HYBRID lexical-only chunk: BM25 score but no semantic score."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        document="syllabus.pdf",
        page=page,
        score=bm25_score,
        chunk_index=0,
        semantic_score=None,
        bm25_score=bm25_score,
    )


def test_hybrid_lexical_only_chunk_requires_bm25_threshold() -> None:
    strong = _lexical_chunk("c1", "cs201 expects two midterms", bm25_score=2.4)
    weak = _lexical_chunk("c2", "the syllabus is short", bm25_score=0.5)

    context = build_context(
        [strong, weak], min_score=0.45, max_chars=2000, min_bm25_score=1.0
    )

    assert [b.chunk_index for b in context.blocks] == [0]  # only the strong hit
    assert context.enough_evidence is True


def test_hybrid_lexical_gate_does_not_apply_without_min_bm25_score() -> None:
    # Without a BM25 gate (NORMAL mode) a lexical-only chunk has no valid score
    # and must be rejected rather than pass the semantic threshold by accident.
    chunk = _lexical_chunk("c1", "cs201 two midterms", bm25_score=2.4)

    context = build_context([chunk], min_score=0.45, max_chars=2000)

    assert context.enough_evidence is False
    assert context.blocks == []


def test_hybrid_semantic_chunk_and_lexical_chunk_coexist() -> None:
    semantic = make_retrieved("attendance is 75 percent", score=0.8, chunk_id="v1")
    lexical = _lexical_chunk("l1", "ENCT 353 requires lab attendance", bm25_score=1.9)

    context = build_context(
        [semantic, lexical], min_score=0.45, max_chars=2000, min_bm25_score=1.0
    )

    assert [b.chunk_index for b in context.blocks] == [0, 0]
    assert context.enough_evidence is True
