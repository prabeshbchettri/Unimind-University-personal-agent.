"""Phase 13 caching tests: embedding memoization and retrieval cache with
index-generation invalidation."""

from __future__ import annotations

import asyncio

from app.caching import EmbeddingCache, RetrievalCache
from app.embedding.cached import CachedEmbedder
from app.retrieval.cached import CachedRetriever
from app.retrieval.models import SearchResult


def _run(coro):
    return asyncio.run(coro)


# --- embedding cache ------------------------------------------------------


class _CountingEmbedder:
    """Minimal embedder that counts how many texts it really embedded."""

    dimensions = 384

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, texts):
        self.calls.extend(texts)
        return [[float(len(text))] * 384 for text in texts]

    def embed_one(self, text: str):
        self.calls.append(text)
        return [float(len(text))] * 384


def test_embedding_cache_serves_hits_without_recomputing() -> None:
    inner = _CountingEmbedder()
    cached = CachedEmbedder(inner, EmbeddingCache(max_entries=10))

    first = cached.embed_one("syllabus")
    second = cached.embed_one("syllabus")

    assert first == second
    assert inner.calls == ["syllabus"]


def test_embedding_cache_batch_misses_only() -> None:
    inner = _CountingEmbedder()
    cached = CachedEmbedder(inner, EmbeddingCache(max_entries=10))

    cached.embed_one("cached text")
    vectors = cached.embed(["cached text", "new text"])

    assert len(vectors) == 2
    assert vectors[0] == [float(len("cached text"))] * 384
    assert inner.calls == ["cached text", "new text"]


def test_embedding_cache_evicts_oldest_entry() -> None:
    inner = _CountingEmbedder()
    cached = CachedEmbedder(inner, EmbeddingCache(max_entries=2))

    cached.embed_one("a")
    cached.embed_one("b")
    cached.embed_one("c")  # evicts "a"
    cached.embed_one("a")  # evicts "b"
    cached.embed_one("b")

    assert inner.calls == ["a", "b", "c", "a", "b"]


# --- retrieval cache ------------------------------------------------------


class _CountingRetriever:
    """Fake retriever counting how many times it actually retrieved."""

    def __init__(self) -> None:
        self.calls = 0

    async def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        self.calls += 1
        return [SearchResult(text=f"{query}-{index}", score=1.0 - index / 10, metadata={}) for index in range(top_k)]


def test_retrieval_cache_hits_until_invalidation() -> None:
    inner = _CountingRetriever()
    cache = RetrievalCache()
    cached = CachedRetriever(inner, cache)

    first = _run(cached.retrieve("attendance", top_k=5))
    second = _run(cached.retrieve("attendance", top_k=5))

    assert first == second
    assert inner.calls == 1

    cache.invalidate()
    _run(cached.retrieve("attendance", top_k=5))
    assert inner.calls == 2


def test_retrieval_cache_ignores_stale_generation_entries() -> None:
    cache = RetrievalCache()
    cache.set("R", "q", 5, [{"text": "old"}], now=100.0)
    assert cache.get("R", "q", 5, now=100.0) == [{"text": "old"}]

    cache.invalidate()
    assert cache.get("R", "q", 5, now=101.0) is None


def test_retrieval_cache_ttl_expires_entries() -> None:
    cache = RetrievalCache(ttl_seconds=60)
    cache.set("R", "q", 5, [{"text": "old"}], now=1000.0)
    assert cache.get("R", "q", 5, now=1059.0) is not None
    assert cache.get("R", "q", 5, now=1060.5) is None


def test_retrieval_cache_keyed_by_query_and_top_k() -> None:
    inner = _CountingRetriever()
    cache = RetrievalCache()
    cached = CachedRetriever(inner, cache)

    _run(cached.retrieve("attendance", top_k=3))
    _run(cached.retrieve("attendance", top_k=5))
    _run(cached.retrieve("grading", top_k=3))

    assert inner.calls == 3


def test_indexing_invalidates_retrieval_cache() -> None:
    from tests import rag_factory

    cache = RetrievalCache()
    service = rag_factory.make_indexing_service(retrieval_cache=cache)
    inner = _CountingRetriever()
    cached = CachedRetriever(inner, cache)

    _run(cached.retrieve("anything", top_k=5))  # miss
    _run(cached.retrieve("anything", top_k=5))  # hit
    assert inner.calls == 1

    _run(service.index(rag_factory.make_structured(["Some syllabus text."], document_id="doc-cache")))
    assert cache.generation == 1

    _run(cached.retrieve("anything", top_k=5))  # stale generation -> miss
    assert inner.calls == 2