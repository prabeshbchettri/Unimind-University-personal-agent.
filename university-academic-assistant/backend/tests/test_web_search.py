"""Tests for web search integration (Phase 11).

Covers the required scenarios:
1. Internal university question stays internal (university priority).
2. Current/external question routes to WEB with preserved sources.
3. Mixed question combines university documents with web results.
4. No useful internal evidence -> explicit insufficiency, never fabrication.
5. Web failure -> graceful fallback to internal retrieval.
6. Search timeout -> same graceful fallback.
"""

import asyncio
import time

import pytest

from app.rag.context import NO_CONTEXT_NOTE
from app.retrieval.models import SearchResult
from app.router.analyzers import RuleBasedQueryAnalyzer
from app.router.eval_set import ROUTING_EVAL_SET
from app.router.plan import RetrievalStrategy
from app.services.chat import ChatService
from app.web.errors import WebSearchError
from app.web.providers import DuckDuckGoWebSearchProvider
from tests import rag_factory as factory


def _run(coro):
    return asyncio.run(coro)


class FakeRetriever:
    """Minimal retriever double with the ``retrieve(query, top_k)`` interface."""

    name = "fake"

    def __init__(self, results=None) -> None:
        self._results = list(results or [])
        self.calls = 0

    async def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        self.calls += 1
        return list(self._results)


class CountingStubProvider(factory.make_web_provider().__class__):
    """Stub provider that records how often it was asked."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def search(self, query: str, top_k: int = 4):
        self.calls += 1
        return await super().search(query, top_k=top_k)


class FailingProvider:
    """Provider that always fails (models an unreachable search backend)."""

    name = "failing"

    async def search(self, query: str, top_k: int = 4):
        raise WebSearchError("provider unavailable")


class SlowProvider:
    """Provider that sleeps far beyond any reasonable timeout."""

    name = "slow"

    async def search(self, query: str, top_k: int = 4):
        await asyncio.sleep(5)
        return []


def _internal_chunk() -> SearchResult:
    return factory.normalization_chunk()


def _make_router(normal_results=None, hybrid_results=None, provider=None, web_timeout=1.0, **kwargs):
    """Adaptive router with fakes for the internal retrievers."""
    return factory.make_adaptive_router(
        normal_retriever=FakeRetriever(normal_results),
        hybrid_retriever=FakeRetriever(hybrid_results),
        top_k=5,
        web_retriever=factory.make_web_retriever(provider=provider, timeout=web_timeout),
        **kwargs,
    )


def test_internal_question_stays_internal_and_never_touches_web() -> None:
    provider = CountingStubProvider()
    router = _make_router(normal_results=[_internal_chunk()], provider=provider)

    plan = router.analyze("What is the attendance requirement?")
    assert plan.strategy == RetrievalStrategy.NORMAL

    results = _run(router.retrieve("What is the attendance requirement?"))
    assert results == [_internal_chunk()]
    assert router.last_plan.strategy == RetrievalStrategy.NORMAL
    assert provider.calls == 0


def test_university_questions_prefer_university_documents() -> None:
    analyzer = RuleBasedQueryAnalyzer()
    assert analyzer.analyze("What is the attendance requirement?").strategy == RetrievalStrategy.NORMAL
    assert analyzer.analyze("When does the academic calendar start?").strategy == RetrievalStrategy.NORMAL
    assert analyzer.analyze("Which subjects are in semester 5?").strategy == RetrievalStrategy.GRAPH
    assert analyzer.analyze("What does regulation 12 say about attendance?").strategy == RetrievalStrategy.HYBRID


def test_current_external_question_routes_to_web_with_preserved_sources() -> None:
    router = _make_router()
    plan = router.analyze("What is the latest Python version?")
    assert plan.strategy == RetrievalStrategy.WEB
    assert plan.web_parameters == {}

    results = _run(router.retrieve("What is the latest Python version?"))
    assert results
    top = results[0]
    assert top.collection == "web"
    assert top.method == "web"
    assert top.metadata["url"].startswith("https://")
    assert top.metadata["title"]
    assert top.metadata["retrieved_at"]
    assert "Python" in top.text
    assert router.last_plan.fallback is False


def test_rule_analyzer_classifies_web_queries() -> None:
    analyzer = RuleBasedQueryAnalyzer()
    for item in [entry for entry in ROUTING_EVAL_SET if entry["expected"] == "WEB"]:
        profile = analyzer.analyze(item["query"])
        assert profile.strategy == RetrievalStrategy.WEB
        assert profile.reason
        assert profile.confidence == 0.9


def test_mixed_question_detected_and_web_supplements_internal() -> None:
    router = _make_router(hybrid_results=[_internal_chunk()])

    plan = router.analyze("What is the current attendance requirement?")
    assert plan.strategy == RetrievalStrategy.WEB
    assert plan.web_parameters.get("mixed") is True

    results = _run(router.retrieve("What is the current attendance requirement?"))
    assert len(results) == 2
    assert results[0].collection == "university_docs"
    assert results[1].collection == "web"
    assert router.last_plan.fallback is False


def test_no_useful_evidence_is_not_confidently_answered() -> None:
    captured: dict = {}
    provider = factory.make_web_provider()

    def responder(prompt: str) -> str:
        captured["prompt"] = prompt
        if NO_CONTEXT_NOTE in prompt:
            return "I could not find sufficient information in the available sources."
        return "Confidently fabricated answer."

    router = _make_router(provider=provider)
    service = ChatService(
        retriever=router,
        context_builder=factory.make_context_builder(),
        llm=factory.make_llm(responder=responder),
        top_k=5,
    )
    result = _run(service.answer("What is the current price of Bitcoin?"))

    assert result.sources == []
    assert result.answer == "I could not find sufficient information in the available sources."
    assert "WEB RESULTS:" not in captured["prompt"]


def test_web_failure_falls_back_to_internal_gracefully() -> None:
    router = _make_router(hybrid_results=[_internal_chunk()], provider=FailingProvider())

    results = _run(router.retrieve("What is the latest Python version?"))
    assert results == [_internal_chunk()]
    assert router.last_plan.fallback is True


def test_web_failure_with_no_internal_evidence_returns_empty() -> None:
    router = _make_router(provider=FailingProvider())

    results = _run(router.retrieve("What is the latest Python version?"))
    assert results == []
    assert router.last_plan.fallback is True


def test_web_timeout_treated_as_failure() -> None:
    router = _make_router(hybrid_results=[_internal_chunk()], provider=SlowProvider(), web_timeout=0.05)

    start = time.monotonic()
    results = _run(router.retrieve("What is the latest Python version?"))
    elapsed = time.monotonic() - start

    assert elapsed < 2.0  # the 5s sleep never blocks the answer
    assert results == [_internal_chunk()]
    assert router.last_plan.fallback is True


def test_chat_service_renders_web_block_with_citations() -> None:
    captured: dict = {}

    def responder(prompt: str) -> str:
        captured["prompt"] = prompt
        return "Python 3.13 is the latest release, per the official downloads page."

    router = _make_router()
    service = ChatService(
        retriever=router,
        context_builder=factory.make_context_builder(),
        llm=factory.make_llm(responder=responder),
        top_k=5,
    )
    result = _run(service.answer("What is the latest Python version?"))

    assert result.answer
    assert result.sources[0].collection == "web"
    assert "WEB RESULTS:" in captured["prompt"]
    assert "[Web 1]" in captured["prompt"]
    assert "https://www.python.org/downloads/" in captured["prompt"]
    assert "retrieved" in captured["prompt"]
    assert "CONTEXT:" in captured["prompt"]


def test_mixed_question_prompt_contains_both_blocks() -> None:
    captured: dict = {}

    def responder(prompt: str) -> str:
        captured["prompt"] = prompt
        return "answer"

    router = _make_router(hybrid_results=[_internal_chunk()])
    service = ChatService(
        retriever=router,
        context_builder=factory.make_context_builder(),
        llm=factory.make_llm(responder=responder),
        top_k=5,
    )
    _run(service.answer("What is the current attendance requirement?"))

    assert "CONTEXT:" in captured["prompt"]
    assert "WEB RESULTS:" in captured["prompt"]
    assert "[Source 1]" in captured["prompt"]
    assert "[Web 1]" in captured["prompt"]


def test_context_builder_renders_web_block_only_when_provided() -> None:
    builder = factory.make_context_builder()
    web_chunk = SearchResult(
        text="Python 3.13 is the latest stable release.",
        score=1.0,
        metadata={
            "title": "Python 3.13 Released",
            "url": "https://www.python.org/downloads/",
            "retrieved_at": "2026-08-13T00:00:00+00:00",
        },
        collection="web",
        method="web",
    )

    _, prompt = builder.build("Python version", [_internal_chunk()])
    assert "WEB RESULTS:" not in prompt

    _, prompt = builder.build("Python version", [_internal_chunk()], web_results=[web_chunk])
    assert "WEB RESULTS:" in prompt
    assert "--- END OF WEB RESULTS ---" in prompt
    assert "[Web 1] Python 3.13 Released (https://www.python.org/downloads/), retrieved 2026-08-13T00:00:00+00:00" in prompt
    assert "Python 3.13 is the latest stable release." in prompt
    # Web sources stay outside the university CONTEXT block.
    assert "[Source 1]" in prompt


def test_system_prompt_grounds_web_safety() -> None:
    from app.rag.context import SYSTEM_PROMPT

    assert "cited with their URL" in SYSTEM_PROMPT
    assert "not automatically trustworthy" in SYSTEM_PROMPT
    assert "university documents take precedence over web results" in SYSTEM_PROMPT


def test_stub_provider_is_deterministic_and_attributed() -> None:
    provider = factory.make_web_provider()
    first = _run(provider.search("What is the latest Python version?"))
    second = _run(provider.search("What is the latest Python version?"))

    assert first == second
    assert first[0].title
    assert first[0].url.startswith("https://")
    assert first[0].snippet
    assert first[0].retrieved_at == second[0].retrieved_at
    assert _run(provider.search("unknown topic with no match")) == []


def test_duckduckgo_provider_parses_html_hermetically() -> None:
    provider = DuckDuckGoWebSearchProvider(timeout_seconds=0.5)
    page = (
        '<html><body><div class="result">'
        '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.python.org%2Fdownloads%2F&amp;rut=abc">'
        "Python 3.13 Released</a>"
        '<a class="result__snippet">The latest stable release with a JIT compiler.</a>'
        "</div></body></html>"
    )
    results = provider._parse_html(page, 2)

    assert len(results) == 1
    assert results[0].title == "Python 3.13 Released"
    assert results[0].url == "https://www.python.org/downloads/"
    assert "JIT compiler" in results[0].snippet
    assert results[0].retrieved_at


def test_web_search_endpoint_returns_results(client) -> None:
    response = client.post("/web/search", json={"query": "What is the latest Python version?"})
    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == "stub"
    assert body["results"]
    assert body["results"][0]["url"] == "https://www.python.org/downloads/"
    assert body["results"][0]["retrieved_at"]


def test_router_plan_endpoint_reports_web_strategy(client) -> None:
    response = client.post("/router/plan", json={"query": "What is the latest Python version?"})
    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "WEB"
    assert "web" in body["reason"].lower()

    internal = client.post("/router/plan", json={"query": "What is the attendance requirement?"}).json()
    assert internal["strategy"] == "NORMAL"


def test_chat_endpoint_distinguishes_web_and_university_sources(client) -> None:
    web_response = client.post("/chat", json={"message": "What is the latest Python version?"})
    assert web_response.status_code == 200
    web_body = web_response.json()
    assert web_body["strategy"] == "WEB"
    assert web_body["sources"]
    top = web_body["sources"][0]
    assert top["kind"] == "web"
    assert top["url"] == "https://www.python.org/downloads/"
    assert top["retrieved_at"]
    assert top["collection"] == "web"
    assert top["retrieval_method"] == "web"

    internal_response = client.post("/chat", json={"message": "What is the attendance requirement?"})
    assert internal_response.status_code == 200
    assert internal_response.json()["strategy"] == "NORMAL"
    for source in internal_response.json()["sources"]:
        assert source["kind"] == "university"
        assert source["url"] == ""
