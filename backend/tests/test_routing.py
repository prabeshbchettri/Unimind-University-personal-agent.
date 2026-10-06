"""Tests for the deterministic query router.

These exercise the real routing rules (regex/lexical features), not mocks:

- simple conceptual queries -> NORMAL
- course codes / acronyms / exact policy terms -> HYBRID
- ambiguous queries with no strong signal -> HYBRID fallback (intentional)
"""

from __future__ import annotations

import pytest

from app.routing import (
    STRATEGY_HYBRID,
    STRATEGY_NORMAL,
    QueryRouter,
)


@pytest.fixture
def router() -> QueryRouter:
    return QueryRouter()


def test_simple_conceptual_question_routes_to_normal(router) -> None:
    decision = router.route("What is the purpose of continuous assessment?")

    assert decision.strategy == STRATEGY_NORMAL
    assert decision.reason  # explainable: reason is populated
    assert decision.confidence == "medium"


def test_factual_number_question_routes_to_normal(router) -> None:
    decision = router.route("How many books can I borrow from the library?")

    assert decision.strategy == STRATEGY_NORMAL
    assert decision.reason


def test_course_code_routes_to_hybrid(router) -> None:
    decision = router.route("What are the attendance requirements for ENCT 353?")

    assert decision.strategy == STRATEGY_HYBRID
    assert decision.confidence == "high"
    assert "course code" in decision.reason


def test_compact_course_code_routes_to_hybrid(router) -> None:
    assert router.route("Is CS201 a prerequisite for anything?").strategy == STRATEGY_HYBRID


def test_exact_policy_terminology_routes_to_hybrid(router) -> None:
    decision = router.route("What is the attendance policy?")

    assert decision.strategy == STRATEGY_HYBRID
    assert "policy" in decision.reason
    assert decision.confidence == "medium"


def test_acronym_routes_to_hybrid(router) -> None:
    decision = router.route("What is the GPA requirement for graduation?")

    assert decision.strategy == STRATEGY_HYBRID
    assert "GPA" in decision.reason


def test_ambiguous_query_defaults_to_hybrid_fallback(router) -> None:
    decision = router.route("Tell me about examinations")

    assert decision.strategy == STRATEGY_HYBRID
    assert decision.confidence == "low"
    assert "defaulting to hybrid" in decision.reason


def test_bare_query_defaults_to_hybrid_fallback(router) -> None:
    assert router.route("Hello").strategy == STRATEGY_HYBRID


def test_multi_part_question_routes_to_hybrid(router) -> None:
    decision = router.route("What are the attendance and examination rules?")

    assert decision.strategy == STRATEGY_HYBRID
    assert "multi-part" in decision.reason


def test_empty_query_is_rejected(router) -> None:
    with pytest.raises(ValueError, match="empty"):
        router.route("   ")