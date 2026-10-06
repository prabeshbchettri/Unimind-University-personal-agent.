"""End-to-end RAG orchestrator tests (fakes for embeddings, store, LLM).

The pipeline wiring is real: retrieve -> build context -> (maybe) generate.
Only the embedding model, the Qdrant engine and the LLM are faked.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.context import build_context
from app.rag import INSUFFICIENT_EVIDENCE_ANSWER, SYSTEM_PROMPT, answer_question
from app.retriever import RetrievedChunk, VectorRetriever
from app.routing import QueryRouter
from tests.conftest import FakeEmbedding, FakeLLM, FakeQdrant, FakeRetriever, make_retrieved
from tests.test_ingest_pipeline import _make_store

ATTENDANCE_TEXT = (
    "Students are required to attend at least 75 percent of all scheduled "
    "sessions in every registered course."
)
EXAM_TEXT = (
    "Eligibility to sit a semester-end examination requires registration in "
    "good standing and settlement of all outstanding fees."
)


def build_rag(monkeypatched_settings: Settings, texts: dict[str, str]) -> tuple:
    """Seed a fake store with texts and return (retriever, llm)."""
    embedding = FakeEmbedding(dim=64)
    client = FakeQdrant(dim=64)
    store = _make_store(client, monkeypatched_settings)
    store.ensure_collection()

    chunks = []
    vectors = []
    for name, text in texts.items():
        chunks.append(_chunk(name, text))
        vectors.append(embedding.embed_query(text))
    store.upsert_chunks(chunks, vectors)

    retriever = VectorRetriever(store=store, embedding=embedding, top_k=4)
    return retriever, embedding


def _chunk(name: str, text: str):
    from app.ingestion import Chunk

    return Chunk(
        id=f"{name}::p1::c0",
        document_name=name,
        source_path=f"data/documents/{name}",
        page=1,
        chunk_index=0,
        text=text,
    )


@pytest.fixture
def rag_settings() -> Settings:
    return Settings(
        qdrant_url="http://fake",
        qdrant_collection="test_docs",
        embedding_dim=64,
        retrieval_top_k=4,
        min_relevance_score=0.0,  # tests control sufficiency via scores/threshold below
        max_context_chars=4000,
        _env_file=None,
    )


def test_known_question_returns_grounded_answer_with_citation(rag_settings) -> None:
    retriever, _ = build_rag(rag_settings, {"attendance_policy.pdf": ATTENDANCE_TEXT})
    llm = FakeLLM(text="The minimum attendance requirement is 75 percent [1].")

    result = answer_question(
        "What is the minimum attendance requirement?",
        settings=rag_settings,
        retriever=retriever,
        llm=llm,
    )

    assert result.enough_evidence is True
    assert "75 percent" in result.answer
    assert result.provider == "fake"
    assert len(result.sources) == 1
    assert result.sources[0].document == "attendance_policy.pdf"
    assert result.sources[0].page == 1


def test_cjk_style_citations_are_normalized(rag_settings) -> None:
    """Groq's gpt-oss-20b sometimes emits full-width brackets like【1】.

    The grounding guarantee must hold regardless of bracket style: the marker
    is normalized to [1] so it survives citation validation.
    """
    retriever, _ = build_rag(rag_settings, {"attendance_policy.pdf": ATTENDANCE_TEXT})
    llm = FakeLLM(text="The minimum is 75 percent【1】.")

    result = answer_question(
        "What is the minimum attendance requirement?",
        settings=rag_settings,
        retriever=retriever,
        llm=llm,
    )

    assert result.enough_evidence is True
    assert "75 percent[1]" in result.answer
    assert "\u3010" not in result.answer  # U+3010 LEFT BLACK LENTICULAR BRACKET
    assert result.retrieved[0].score >= 0  # metadata present

    # The LLM was prompted with the grounding rules and the evidence.
    call = llm.calls[0]
    assert call["system_prompt"] == SYSTEM_PROMPT
    assert "75 percent" in call["context"]
    assert "attendance_policy.pdf" in call["context"]


def test_api_shape_contains_answer_sources_and_retrieval_info(rag_settings) -> None:
    retriever, _ = build_rag(
        rag_settings, {"attendance_policy.pdf": ATTENDANCE_TEXT, "exams.pdf": EXAM_TEXT}
    )
    llm = FakeLLM(text="Answer [1].")

    body = answer_question(
        "attendance rules",
        settings=rag_settings,
        retriever=retriever,
        llm=llm,
    ).to_api_dict()

    assert set(body) == {
        "answer", "sources", "enough_evidence", "retrieval",
        "provider", "model", "strategy", "strategy_reason",
    }
    assert body["strategy"] == "normal"  # no router injected -> default strategy
    assert "Router not configured" in body["strategy_reason"]
    assert body["sources"][0] == {"document": "attendance_policy.pdf", "page": 1}
    assert body["retrieval"]["chunks_retrieved"] >= 1
    assert body["retrieval"]["chunks_used"] >= 1


def test_irrelevant_question_returns_controlled_insufficient_evidence(rag_settings) -> None:
    retriever, _ = build_rag(rag_settings, {"attendance_policy.pdf": ATTENDANCE_TEXT})
    llm = FakeLLM()
    settings = rag_settings.model_copy(update={"min_relevance_score": 0.99})

    result = answer_question(
        "Who won the football world cup?",
        settings=settings,
        retriever=retriever,
        llm=llm,
    )

    assert result.enough_evidence is False
    assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert result.sources == []
    assert result.provider == "none"  # no generation happened
    assert llm.calls == []  # the LLM is never called without evidence


def test_context_limit_is_respected(rag_settings) -> None:
    retriever, _ = build_rag(
        rag_settings,
        {"a.pdf": ATTENDANCE_TEXT, "b.pdf": EXAM_TEXT, "c.pdf": "Library loans: six books."},
    )
    llm = FakeLLM()
    settings = rag_settings.model_copy(update={"max_context_chars": 120})

    result = answer_question(
        "attendance and exams and library",
        settings=settings,
        retriever=retriever,
        llm=llm,
    )

    assert result.enough_evidence is True
    used_context = llm.calls[0]["context"]
    # Evidence text minus the rendered "[n] source:" headers stays within budget.
    evidence_chars = sum(len(block.text) for block in build_context(
        result.retrieved, min_score=settings.min_relevance_score, max_chars=settings.max_context_chars
    ).blocks)
    assert evidence_chars <= 120
    assert len(used_context) < 400  # rendering overhead is bounded too


def test_unsupported_citations_are_stripped(rag_settings) -> None:
    retriever, _ = build_rag(rag_settings, {"attendance_policy.pdf": ATTENDANCE_TEXT})
    llm = FakeLLM(text="Answer cites [1] and invents [9].")

    result = answer_question(
        "attendance", settings=rag_settings, retriever=retriever, llm=llm
    )

    assert "[1]" in result.answer
    assert "[9]" not in result.answer


def test_empty_query_is_rejected(rag_settings) -> None:
    retriever, _ = build_rag(rag_settings, {"attendance_policy.pdf": ATTENDANCE_TEXT})
    with pytest.raises(ValueError, match="empty"):
        answer_question("   ", settings=rag_settings, retriever=retriever, llm=FakeLLM())


# ---------------------------------------------------------------------------
# Phase 4: adaptive retrieval (router + strategy dispatch)
# ---------------------------------------------------------------------------


def _lexical_only(text: str, bm25_score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="lex::p1::c0",
        text=text,
        document="syllabus.pdf",
        page=1,
        score=bm25_score,
        chunk_index=0,
        semantic_score=None,
        bm25_score=bm25_score,
    )


def test_router_dispatches_simple_query_to_normal_retriever(rag_settings) -> None:
    llm = FakeLLM(text="Continuous assessment is coursework [1].")
    normal = FakeRetriever(
        [make_retrieved("continuous assessment contributes to the grade", score=0.8)]
    )
    hybrid = FakeRetriever([])
    router = QueryRouter()

    result = answer_question(
        "What is the purpose of continuous assessment?",
        settings=rag_settings,
        retriever=normal,
        llm=llm,
        router=router,
        hybrid_retriever=hybrid,
    )

    assert result.strategy == "normal"
    assert result.strategy_reason
    assert result.enough_evidence is True
    assert normal.queries == ["What is the purpose of continuous assessment?"]
    assert hybrid.queries == []  # hybrid strategy was not invoked


def test_router_dispatches_course_code_query_to_hybrid_retriever(rag_settings) -> None:
    llm = FakeLLM(text="CS201 is four credits [1].")
    normal = FakeRetriever([])
    hybrid = FakeRetriever(
        [make_retrieved("CS201 is a four credit core course", document="cs201_syllabus.pdf", score=0.7)]
    )
    router = QueryRouter()

    result = answer_question(
        "What are the credit requirements for CS201?",
        settings=rag_settings,
        retriever=normal,
        llm=llm,
        router=router,
        hybrid_retriever=hybrid,
    )

    assert result.strategy == "hybrid"
    assert result.enough_evidence is True
    assert hybrid.queries == ["What are the credit requirements for CS201?"]
    assert normal.queries == []


def test_router_uncertain_query_falls_back_to_hybrid(rag_settings) -> None:
    llm = FakeLLM()
    normal = FakeRetriever([])
    hybrid = FakeRetriever([make_retrieved("some evidence", score=0.8)])
    router = QueryRouter()

    result = answer_question(
        "Tell me about examinations",
        settings=rag_settings,
        retriever=normal,
        llm=llm,
        router=router,
        hybrid_retriever=hybrid,
    )

    assert result.strategy == "hybrid"
    assert hybrid.queries == ["Tell me about examinations"]


def test_hybrid_insufficient_evidence_is_controlled_and_no_llm(rag_settings) -> None:
    llm = FakeLLM()
    normal = FakeRetriever([])
    hybrid = FakeRetriever([make_retrieved("weak match", score=0.2)])
    settings = rag_settings.model_copy(update={"min_relevance_score": 0.99})
    router = QueryRouter()

    result = answer_question(
        "football transfer window rules",
        settings=settings,
        retriever=normal,
        llm=llm,
        router=router,
        hybrid_retriever=hybrid,
    )

    assert result.strategy == "hybrid"
    assert result.enough_evidence is False
    assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert result.sources == []
    assert llm.calls == []


def test_hybrid_lexical_only_evidence_is_gated_by_bm25_threshold(rag_settings) -> None:
    llm = FakeLLM()
    normal = FakeRetriever([])
    # Only a weak lexical-only chunk: below MIN_BM25_SCORE, so no evidence.
    hybrid = FakeRetriever([_lexical_only("cs201 two midterms", bm25_score=0.4)])
    settings = rag_settings.model_copy(
        update={"min_relevance_score": 0.45, "min_bm25_score": 1.0}
    )
    router = QueryRouter()

    result = answer_question(
        "What assessments does CS201 have?",
        settings=settings,
        retriever=normal,
        llm=llm,
        router=router,
        hybrid_retriever=hybrid,
    )

    assert result.enough_evidence is False
    assert llm.calls == []


def test_hybrid_strong_lexical_only_evidence_enters_context(rag_settings) -> None:
    llm = FakeLLM(text="Two midterms worth 20 percent each [1].")
    normal = FakeRetriever([])
    hybrid = FakeRetriever([_lexical_only("cs201 has two midterms worth twenty percent", bm25_score=2.2)])
    settings = rag_settings.model_copy(
        update={"min_relevance_score": 0.45, "min_bm25_score": 1.0}
    )
    router = QueryRouter()

    result = answer_question(
        "How many midterms does CS201 have?",
        settings=settings,
        retriever=normal,
        llm=llm,
        router=router,
        hybrid_retriever=hybrid,
    )

    assert result.enough_evidence is True
    assert len(result.sources) == 1
    assert result.sources[0].document == "syllabus.pdf"
    assert llm.calls[0]["context"].startswith("[1] source:")
