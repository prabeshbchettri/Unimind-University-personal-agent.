"""Tests for the query analyzer and adaptive router (Phase 8)."""

import asyncio

import pytest

from app.graph.extractor import GraphExtractor
from app.llm.stub import StubLLMClient
from app.repositories.graph import InMemoryGraphRepository
from app.retrieval.graph_retriever import GRAPH_COLLECTION, GraphRetriever
from app.retrieval.models import SearchResult
from app.router.analyzers import LLMQueryAnalyzer, RuleBasedQueryAnalyzer
from app.router.eval_set import ROUTING_EVAL_SET, evaluate_router
from app.router.plan import RetrievalStrategy
from app.services.graph import KnowledgeGraphService
from tests.rag_factory import (
    make_adaptive_router,
    make_graph_past_paper,
    make_graph_retriever,
    make_graph_syllabus,
    make_rule_analyzer,
    normalization_chunk,
)


def _run(coro):
    return asyncio.run(coro)


class FakeRetriever:
    """Minimal retriever double with the ``retrieve(query, top_k)`` interface."""

    name = "fake"

    def __init__(self, results=None) -> None:
        self._results = list(results or [])

    async def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        return list(self._results)


def _make_router(**kwargs):
    return make_adaptive_router(
        normal_retriever=FakeRetriever(),
        hybrid_retriever=FakeRetriever(),
        graph_retriever=make_graph_retriever(make_graph_service()),
        top_k=5,
        **kwargs,
    )


def make_graph_service() -> KnowledgeGraphService:
    service = KnowledgeGraphService(repository=InMemoryGraphRepository(), extractor=GraphExtractor())
    _run(service.index(make_graph_syllabus(document_id="doc-syllabus")))
    _run(service.index(make_graph_past_paper(document_id="doc-past")))
    return service


# ---------------------------------------------------------------------------
# Rule-based analyzer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query",
    [item["query"] for item in ROUTING_EVAL_SET if item["expected"] == "GRAPH"],
)
def test_rule_analyzer_classifies_graph_queries(query: str) -> None:
    profile = make_rule_analyzer().analyze(query)
    assert profile.strategy == RetrievalStrategy.GRAPH
    assert profile.reason
    assert profile.graph_parameters.get("intent")


@pytest.mark.parametrize(
    "query",
    [item["query"] for item in ROUTING_EVAL_SET if item["expected"] == "HYBRID"],
)
def test_rule_analyzer_classifies_hybrid_queries(query: str) -> None:
    profile = make_rule_analyzer().analyze(query)
    assert profile.strategy == RetrievalStrategy.HYBRID
    assert profile.reason


@pytest.mark.parametrize(
    "query",
    [item["query"] for item in ROUTING_EVAL_SET if item["expected"] == "NORMAL"],
)
def test_rule_analyzer_classifies_normal_queries(query: str) -> None:
    profile = make_rule_analyzer().analyze(query)
    assert profile.strategy == RetrievalStrategy.NORMAL
    assert profile.confidence == 0.7


def test_rule_analyzer_regulation_query_sets_filter() -> None:
    profile = make_rule_analyzer().analyze("What does regulation 12 say about attendance?")
    assert profile.strategy == RetrievalStrategy.HYBRID
    assert profile.filters == {"document_type": "rules_regulations"}


def test_rule_analyzer_empty_query_has_zero_confidence() -> None:
    profile = make_rule_analyzer().analyze("   ")
    assert profile.confidence == 0.0
    assert profile.strategy is None


def test_rule_analyzer_question_number_not_misclassified_as_graph() -> None:
    profile = make_rule_analyzer().analyze("What is question 5 about?")
    assert profile.strategy == RetrievalStrategy.HYBRID


# ---------------------------------------------------------------------------
# LLM analyzer
# ---------------------------------------------------------------------------

def test_llm_analyzer_returns_validated_strategy() -> None:
    llm = StubLLMClient(responder=lambda prompt: '{"strategy": "GRAPH", "reason": "topics of a subject"}')
    profile = LLMQueryAnalyzer(llm).analyze("List topics of DBMS.")
    assert profile.strategy == RetrievalStrategy.GRAPH
    assert profile.confidence == 0.5


def test_llm_analyzer_invalid_json_is_unknown() -> None:
    llm = StubLLMClient(responder=lambda prompt: "not json")
    profile = LLMQueryAnalyzer(llm).analyze("List topics of DBMS.")
    assert profile.confidence == 0.0


def test_llm_analyzer_invalid_strategy_is_unknown() -> None:
    llm = StubLLMClient(responder=lambda prompt: '{"strategy": "TURBO"}')
    profile = LLMQueryAnalyzer(llm).analyze("List topics of DBMS.")
    assert profile.confidence == 0.0


def test_llm_analyzer_empty_query_is_unknown() -> None:
    profile = LLMQueryAnalyzer(StubLLMClient()).analyze("")
    assert profile.confidence == 0.0


# ---------------------------------------------------------------------------
# Graph retriever
# ---------------------------------------------------------------------------

def test_graph_retriever_retrieves_subjects_in_semester() -> None:
    retriever = make_graph_retriever(make_graph_service())
    results = _run(retriever.retrieve("Which subjects are in semester 5?"))
    assert results
    for result in results:
        assert result.collection == GRAPH_COLLECTION
        assert result.method == "graph"
        assert result.metadata["semester"] == 5
    assert any("Database Management System" in result.text for result in results)


def test_graph_retriever_retrieves_questions_about_topic() -> None:
    retriever = make_graph_retriever(make_graph_service())
    results = _run(retriever.retrieve("Which past questions are about Normalization?"))
    assert results
    assert any(result.metadata["intent"] == "questions_about_topic" for result in results)


def test_graph_retriever_non_graph_query_returns_empty() -> None:
    retriever = make_graph_retriever(make_graph_service())
    results = _run(retriever.retrieve("Explain normalization."))
    assert results == []


def test_graph_retriever_unknown_entity_falls_back_to_dense() -> None:
    dense = FakeRetriever([normalization_chunk()])
    retriever = make_graph_retriever(make_graph_service(), dense_retriever=dense)
    results = _run(retriever.retrieve("Which subjects are in semester 12?"))
    assert results == [normalization_chunk()]


def test_graph_retriever_empty_graph_returns_no_results() -> None:
    retriever = make_graph_retriever(make_graph_service())
    results = _run(retriever.retrieve("Which subjects are in semester 9?"))
    assert results == []


# ---------------------------------------------------------------------------
# Adaptive router
# ---------------------------------------------------------------------------

def test_adaptive_router_returns_plan_for_each_strategy() -> None:
    router = _make_router()
    cases = {
        "Explain normalization.": RetrievalStrategy.NORMAL,
        "What does regulation 12 say about attendance?": RetrievalStrategy.HYBRID,
        "Which subjects are in semester 5?": RetrievalStrategy.GRAPH,
    }
    for query, expected in cases.items():
        plan = router.analyze(query)
        assert plan.strategy == expected
        assert plan.query == query
        assert plan.reason


def test_adaptive_router_executes_selected_strategy() -> None:
    router = _make_router()
    _run(router.retrieve("Which subjects are in semester 5?"))
    assert router.last_plan.strategy == RetrievalStrategy.GRAPH
    _run(router.retrieve("Explain normalization."))
    assert router.last_plan.strategy == RetrievalStrategy.NORMAL
    _run(router.retrieve("What does regulation 12 say about attendance?"))
    assert router.last_plan.strategy == RetrievalStrategy.HYBRID


def test_adaptive_router_empty_query_uses_fallback() -> None:
    router = _make_router()
    plan = router.analyze("")
    assert plan.fallback is True
    assert plan.strategy == RetrievalStrategy.HYBRID


def test_adaptive_router_fixed_strategy_ignores_classification() -> None:
    router = _make_router(fixed_strategy="hybrid")
    plan = router.analyze("Which subjects are in semester 5?")
    assert plan.strategy == RetrievalStrategy.HYBRID


def test_adaptive_router_llm_analyzer_used_for_unknown() -> None:
    llm = StubLLMClient(responder=lambda prompt: '{"strategy": "GRAPH", "reason": "llm said so"}')
    router = _make_router(llm_analyzer=LLMQueryAnalyzer(llm), classifier_mode="llm")
    plan = router.analyze("What is normalization?")
    assert plan.strategy == RetrievalStrategy.GRAPH


def test_adaptive_router_applies_plan_filter() -> None:
    regulation_chunk = SearchResult(
        text="Regulation 12: 75% attendance required.",
        score=0.9,
        metadata={"document_type": "rules_regulations", "title": "Exam Regulations"},
        collection="rules",
    )
    other_chunk = SearchResult(text="Attendance policy.", score=0.8, metadata={"document_type": "syllabus"})
    router = make_adaptive_router(
        normal_retriever=FakeRetriever(),
        hybrid_retriever=FakeRetriever([regulation_chunk, other_chunk]),
        graph_retriever=make_graph_retriever(make_graph_service()),
        top_k=5,
    )
    results = _run(router.retrieve("What does regulation 12 say about attendance?"))
    assert router.last_plan.filters == {"document_type": "rules_regulations"}
    assert results == [regulation_chunk]


def test_adaptive_router_filter_never_empties_results() -> None:
    other_chunk = SearchResult(text="Attendance policy.", score=0.8, metadata={"document_type": "syllabus"})
    router = make_adaptive_router(
        normal_retriever=FakeRetriever(),
        hybrid_retriever=FakeRetriever([other_chunk]),
        graph_retriever=make_graph_retriever(make_graph_service()),
        top_k=5,
    )
    results = _run(router.retrieve("What does regulation 12 say about attendance?"))
    assert results == [other_chunk]


def test_adaptive_router_health_reports_components() -> None:
    health = _run(_make_router().health())
    assert health["name"] == "AdaptiveRouter"
    assert health["status"] == "ok"
    assert health["strategies"] == ["NORMAL", "HYBRID", "GRAPH"]


# ---------------------------------------------------------------------------
# Eval set
# ---------------------------------------------------------------------------

def test_eval_set_has_balanced_categories() -> None:
    counts = {expected: 0 for expected in {"NORMAL", "HYBRID", "GRAPH", "WEB"}}
    for item in ROUTING_EVAL_SET:
        counts[item["expected"]] += 1
    assert all(count >= 6 for count in counts.values())


def test_eval_metrics_are_all_correct() -> None:
    metrics = evaluate_router(_make_router())
    assert metrics["total"] == 36
    assert metrics["routing_accuracy"] == 1.0
    assert metrics["false_routing_rate"] == 0.0
    assert metrics["fallback_rate"] == 0.0


# ---------------------------------------------------------------------------
# API integration
# ---------------------------------------------------------------------------

def test_router_plan_endpoint_returns_structured_plan(client) -> None:
    response = client.post("/router/plan", json={"query": "Which subjects are in semester 5?"})
    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "GRAPH"
    assert body["graph_parameters"]["intent"] == "subjects_in_semester"
    assert body["graph_parameters"]["semester"] == 5


def test_router_plan_endpoint_hybrid_filter(client) -> None:
    response = client.post("/router/plan", json={"query": "What does regulation 12 say about attendance?"})
    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "HYBRID"
    assert body["filters"] == {"document_type": "rules_regulations"}


def test_chat_reports_selected_strategy(client) -> None:
    response = client.post("/chat", json={"message": "Explain normalization.", "top_k": 3})
    assert response.status_code == 200
    assert response.json()["strategy"] in {"NORMAL", "HYBRID", "GRAPH"}


def test_chat_graph_question_returns_graph_sources(client, make_text_pdf, tmp_path) -> None:
    path = make_text_pdf(
        tmp_path / "chat_graph_syllabus.pdf",
        [
            "Database Management System Syllabus\n"
            "Semester 5\n"
            "Unit 1: Introduction\nDatabases store data.\n"
            "Unit 2: Normalization\nNormalization removes redundancy in tables.",
        ],
    )
    with open(path, "rb") as handle:
        index = client.post(
            "/documents/index",
            files={"file": ("chat_graph_syllabus.pdf", handle, "application/pdf")},
        )
    assert index.status_code == 200

    response = client.post("/chat", json={"message": "Which subjects are in semester 5?"})
    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "GRAPH"
    assert body["sources"]
    assert body["sources"][0]["collection"] == "knowledge_graph"
