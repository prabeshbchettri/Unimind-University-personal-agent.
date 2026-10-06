"""Generation-layer behavior tests (prompt contract + greeting handling).

The live response style itself is evaluated manually (see
scripts/capture_examples.py); these tests pin the deterministic parts of the
generation contract: what the model is told, how the user turn is framed, and
which requests must never reach the LLM.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.llm import build_user_content
from app.rag import (
    GREETING_ANSWER,
    INSUFFICIENT_EVIDENCE_ANSWER,
    SYSTEM_PROMPT,
    answer_question,
    is_greeting,
)
from tests.conftest import FakeLLM, FakeRetriever, make_retrieved
from tests.test_rag import build_rag


@pytest.fixture
def rag_settings() -> Settings:
    """Same base settings as the RAG orchestrator tests."""
    return Settings(
        qdrant_url="http://fake",
        qdrant_collection="test_docs",
        embedding_dim=64,
        retrieval_top_k=4,
        min_relevance_score=0.0,  # tests control sufficiency via scores/threshold below
        max_context_chars=4000,
        _env_file=None,
    )


# ---------------------------------------------------------------------------
# System prompt contract: the shared prompt that drives Groq AND Ollama.
# ---------------------------------------------------------------------------


def test_system_prompt_groundedness_rules() -> None:
    """The prompt must forbid outside knowledge and invented citations."""
    assert "ONLY" in SYSTEM_PROMPT
    assert "Never use outside knowledge" in SYSTEM_PROMPT
    assert "never invent" in SYSTEM_PROMPT.lower()


def test_system_prompt_requires_natural_intent_aware_answers() -> None:
    """The prompt must demand synthesis over chunk reproduction and intent fit."""
    lowered = SYSTEM_PROMPT.lower()
    assert "your own words" in lowered  # no raw chunk copying
    assert "synthesis" in lowered  # summaries are synthesized, not pasted
    assert "markdown" in lowered  # structure when the request needs it
    for intent in ("definition", "summary", "notes", "comparison", "syllabus"):
        assert intent in lowered  # user intents the style must adapt to


def test_system_prompt_requires_length_control_and_citations() -> None:
    lowered = SYSTEM_PROMPT.lower()
    assert "respect it" in lowered  # user-requested lengths win
    assert "[1]" in SYSTEM_PROMPT and "[2]" in SYSTEM_PROMPT  # citation markers
    assert "directly after each claim" in lowered  # inline citations


def test_system_prompt_abstention_text_matches_controlled_answer() -> None:
    """The model's decline must be identical to the gate's controlled message."""
    collapsed = " ".join(SYSTEM_PROMPT.split())
    assert INSUFFICIENT_EVIDENCE_ANSWER in collapsed
    # No internal retrieval terminology may leak to the user.
    lowered = INSUFFICIENT_EVIDENCE_ANSWER.lower()
    for term in ("chunk", "bm25", "vector", "gate", "retriev", "rrf"):
        assert term not in lowered


def test_user_turn_frames_evidence_as_sources_and_task_last() -> None:
    content = build_user_content("[1] source: a.pdf, page 1\nChapter 1 is Introduction.", "What is Chapter 1?")
    assert content.index("Chapter 1 is Introduction.") < content.index("Question:")
    assert "cited as [1]" in content  # evidence = citable sources
    assert "answer the question the user asked" in content  # task framing
    assert "cite the sources you use" in content


def test_system_prompt_and_user_turn_are_provider_independent_contract() -> None:
    """Groq and Ollama share one prompt pipeline; the RAG layer has no provider branches."""
    import inspect

    from app.llm import GroqLLMClient, OllamaLLMClient

    for client_class in (GroqLLMClient, OllamaLLMClient):
        # Every provider routes its user turn through the same shared builder
        # (directly, or via its payload helper).
        source = inspect.getsource(client_class) if client_class is OllamaLLMClient else inspect.getsource(
            client_class.generate
        )
        assert "build_user_content(" in source, client_class.__name__


# ---------------------------------------------------------------------------
# Greeting handling (pre-RAG conversational check).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "hello",
        "Hello",
        "HI",
        "hey",
        "hello!",
        "hi there",  # harmless extra words, still no factual content
        "Good morning",
        "  hey!  ",
    ],
)
def test_is_greeting_matches_casual_greetings(query: str) -> None:
    assert is_greeting(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "What is the attendance rule?",
        "hi, what is the attendance rule?",  # greeting + real question
        "hey cs201",  # course content hiding behind a greeting word
        "history of simulation",  # substring 'hi'
        "How does Monte Carlo simulation work?",
    ],
)
def test_is_greeting_rejects_questions(query: str) -> None:
    assert is_greeting(query) is False


def test_greeting_is_answered_without_retrieval_or_llm(rag_settings) -> None:
    retriever = FakeRetriever([make_retrieved("evidence", score=0.9)])
    llm = FakeLLM()

    result = answer_question(
        "Hello",
        settings=rag_settings,
        retriever=retriever,
        llm=llm,
        router=None,
        hybrid_retriever=None,
    )

    assert result.answer == GREETING_ANSWER
    assert result.enough_evidence is True
    assert result.provider == "none"  # no generation happened
    assert result.retrieved == []  # no retrieval happened
    assert llm.calls == [] and retriever.queries == []
    assert result.strategy == "greeting"


def test_greeting_with_router_configured_still_skips_rag(rag_settings) -> None:
    retriever = FakeRetriever([make_retrieved("evidence", score=0.9)])
    hybrid = FakeRetriever([make_retrieved("evidence", score=0.9)])
    llm = FakeLLM()

    result = answer_question(
        "hey!",
        settings=rag_settings,
        retriever=retriever,
        llm=llm,
        router=object(),  # router present, as in production
        hybrid_retriever=hybrid,
    )

    assert result.answer == GREETING_ANSWER
    assert llm.calls == [] and retriever.queries == [] and hybrid.queries == []


# ---------------------------------------------------------------------------
# Deterministic abstention and citation guarantees.
# ---------------------------------------------------------------------------


def test_unsupported_question_declines_naturally_without_llm(rag_settings) -> None:
    """No hallucination: the controlled decline fires and mentions nothing internal."""
    retriever, _ = build_rag(rag_settings, {"attendance_policy.pdf": "75 percent rule."})
    llm = FakeLLM()
    settings = rag_settings.model_copy(update={"min_relevance_score": 0.99})

    result = answer_question(
        "Who won the football World Cup?",
        settings=settings,
        retriever=retriever,
        llm=llm,
    )

    assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert result.enough_evidence is False
    assert result.sources == []
    assert llm.calls == []


def test_decline_text_is_natural_and_offers_a_next_step(rag_settings) -> None:
    """The abstention must read like an assistant, not a system error."""
    assert "couldn't find" in INSUFFICIENT_EVIDENCE_ANSWER
    assert "ask me about" in INSUFFICIENT_EVIDENCE_ANSWER


def test_grounded_answer_keeps_inline_citations(rag_settings) -> None:
    """Citation markers pointing at real evidence survive post-processing."""
    retriever, _ = build_rag(rag_settings, {"attendance_policy.pdf": "75 percent rule."})
    llm = FakeLLM(text="Sure! The minimum attendance is 75 percent [1].")

    result = answer_question(
        "What is the minimum attendance?",
        settings=rag_settings,
        retriever=retriever,
        llm=llm,
    )

    assert result.enough_evidence is True
    assert "75 percent [1]" in result.answer
    assert result.sources[0].document == "attendance_policy.pdf"
