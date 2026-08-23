"""Tests for the end-to-end chat pipeline (Phase 5).

Covers the complete vertical slice: query -> retrieval -> context builder ->
LLM -> grounded answer with sources, including the acceptance criteria that
questions with no matching university content are not confidently answered.
"""

import asyncio

from tests import rag_factory as factory

from app.rag.context import NO_CONTEXT_NOTE
from app.services.chat import ChatService

NO_EVIDENCE_ANSWER = "I could not find sufficient information in the available university sources."


def _run(coro):
    return asyncio.run(coro)


def _responder(prompt: str) -> str:
    """Deterministic stand-in for Llama 3.1 8B following the grounding rules."""
    if NO_CONTEXT_NOTE in prompt:
        return NO_EVIDENCE_ANSWER
    return "Normalization removes redundancy in tables."


def _chat_service(**kwargs) -> ChatService:
    service = factory.make_indexing_service()
    factory.index(
        service,
        factory.sample_structured(
            "Unit 1: Introduction\nDBMS is a database management system. Databases store data.\n"
            "Unit 2: Normalization\nNormalization removes redundancy in tables.\n"
            "Unit 3: Transactions\nACID properties guarantee reliable processing."
        ),
    )
    retriever = factory.make_retriever(service.embedder, service.repository, top_k=3)
    return ChatService(
        retriever=retriever,
        context_builder=factory.make_context_builder(),
        llm=factory.make_llm(responder=_responder),
        top_k=3,
    )


def test_answer_is_grounded_in_sources() -> None:
    service = _chat_service()
    result = _run(service.answer("Explain normalization."))

    assert result.answer == "Normalization removes redundancy in tables."
    assert result.sources
    top = result.sources[0]
    assert top.metadata["topic"] == "Normalization"
    assert top.collection == "university_docs"
    assert top.score > 0


def test_question_not_in_database_is_not_confidently_answered() -> None:
    service = _chat_service()
    result = _run(service.answer("What is the capital of France?"))

    # No supporting evidence -> no sources and an explicit insufficiency answer.
    assert result.sources == []
    assert result.answer == NO_EVIDENCE_ANSWER
    assert "could not find sufficient information" in result.answer.lower()


def test_answer_top_k_limits_sources() -> None:
    service = _chat_service()
    result = _run(service.answer("Explain normalization.", top_k=2))
    assert len(result.sources) <= 2


def test_health_reports_pipeline_components() -> None:
    service = _chat_service()
    health = _run(service.health())
    assert health["status"] == "ok"
    assert health["retriever"] == "NormalRetriever"
    assert health["context_builder"] == "ContextBuilder"
    assert health["llm"] == "stub"


def test_answer_empty_message_returns_no_sources() -> None:
    service = _chat_service()
    result = _run(service.answer("   "))
    assert result.sources == []
    assert result.answer == NO_EVIDENCE_ANSWER