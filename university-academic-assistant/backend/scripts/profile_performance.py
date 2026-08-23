"""Phase 13 performance profiling (Part 2).

Measures the pipeline stages on the hermetic environment and prints a
report. The rule for this phase: optimize ONLY where the profile shows a
bottleneck, then re-profile to confirm the improvement.

Stages profiled (in milliseconds):

- chunking      : SemanticChunker over the evaluation corpus
- embedding     : DeterministicEmbedder batch vs per-item
- index_upsert  : full VectorIndexingService.index per corpus document
- retrieval     : normal / hybrid / graph strategy per corpus query
- chat_answer   : full chat pipeline (router + retrieval + context + stub LLM)
- recommend     : full recommendation pipeline

Run from the backend directory:

    python scripts/profile_performance.py

Set the same hermetic environment variables as scripts/evaluate.py, or rely
on the defaults below (the script forces them when unset).
"""

from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QDRANT_URL", ":memory:")
os.environ.setdefault("EMBEDDER_BACKEND", "deterministic")
os.environ.setdefault("LLM_BACKEND", "stub")
os.environ.setdefault("ROUTER_CLASSIFIER", "rules")
os.environ.setdefault("GRAPH_BACKEND", "memory")
os.environ.setdefault("DATABASE_BACKEND", "memory")

from app.evaluation.corpus import DOCUMENTS, RETRIEVAL_QUERIES  # noqa: E402
from app.evaluation.environment import build_environment  # noqa: E402
from app.evaluation.retrieval import _retrieve_docs  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _summary(samples: list[float]) -> dict:
    if not samples:
        return {"count": 0}
    samples = sorted(samples)
    median = statistics.median(samples)
    p95 = samples[min(len(samples) - 1, int(0.95 * len(samples)))]
    return {
        "count": len(samples),
        "mean_ms": round(statistics.mean(samples), 2),
        "median_ms": round(median, 2),
        "p95_ms": round(p95, 2),
        "min_ms": round(samples[0], 2),
        "max_ms": round(samples[-1], 2),
    }


def _bench(fn, repeats: int = 3) -> dict:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return _summary(samples)


def main() -> None:
    env = build_environment()
    env.llm = None  # stub LLM is instant; measure pipeline overhead only

    line = "=" * 72
    print(line)
    print("PHASE 13 - PERFORMANCE PROFILE (hermetic, deterministic)")
    print("machine: Windows x64 | in-memory Qdrant | deterministic embedder")
    print(line)

    # --- chunking ----------------------------------------------------------
    chunker = env.indexing_service.chunker
    text_scale = sum(len(page.text) for doc in DOCUMENTS for page in doc.pages)
    print(f"\ncorpus: {len(DOCUMENTS)} documents, ~{text_scale} chars")

    def chunk_all():
        for doc in DOCUMENTS:
            chunker.chunk(doc)

    print("\n[chunking] SemanticChunker over the full corpus")
    for key, value in _bench(chunk_all, repeats=3).items():
        if key != "count":
            print(f"   {key:<10}: {value}")

    # --- embedding ---------------------------------------------------------
    texts = [chunk.text for doc in DOCUMENTS for chunk in chunker.chunk(doc)]

    def embed_batch():
        env.embedder.embed(texts)

    def embed_one_at_a_time():
        for text in texts:
            env.embedder.embed_one(text)

    print(f"\n[embedding] {len(texts)} chunks, dimensions={env.embedder.dimensions}")
    for label, fn in [("batch  ", embed_batch), ("one-by-one", embed_one_at_a_time)]:
        summary = _bench(fn, repeats=3)
        print(
            f"   {label}  : {summary['mean_ms']:.2f} ms mean, "
            f"{summary['median_ms']:.2f} median, {summary['p95_ms']:.2f} p95 "
            f"({summary['count']} runs)"
        )

    # --- indexing ----------------------------------------------------------
    def index_all():
        for doc in DOCUMENTS:
            _run(env.indexing_service.index(doc))

    print("\n[index] VectorIndexingService.index per corpus document")
    for key, value in _bench(index_all, repeats=2).items():
        if key != "count":
            print(f"   {key:<10}: {value}")

    # --- retrieval ---------------------------------------------------------
    print("\n[retrieval] per query, mean over labeled corpus queries")
    strategies = {
        "NORMAL": env.normal_retriever,
        "HYBRID": env.hybrid_retriever,
        "GRAPH": env.graph_retriever,
    }
    query_count = len(RETRIEVAL_QUERIES)
    for name, retriever in strategies.items():
        def run_queries(retriever=retriever):
            for item in RETRIEVAL_QUERIES:
                _retrieve_docs(retriever, item["query"], top_k=5)

        summary = _bench(run_queries, repeats=3)
        per_query = summary["mean_ms"] / query_count
        print(
            f"   {name:<7}: {per_query:.2f} ms mean per query "
            f"({summary['mean_ms']:.0f} ms for {query_count} queries)"
        )

    # --- chat pipeline -----------------------------------------------------
    print("\n[chat] full answer() pipeline per query (stub LLM)")
    def chat_queries():
        for item in RETRIEVAL_QUERIES:
            _run(env.chat_service.answer(item["query"], top_k=5))

    summary = _bench(chat_queries, repeats=3)
    per_query = summary["mean_ms"] / query_count
    print(
        f"   mean {per_query:.2f} ms per query, "
        f"median {summary['median_ms'] / query_count:.2f} ms, "
        f"p95 {summary['p95_ms'] / query_count:.2f} ms"
    )

    # --- recommendation ----------------------------------------------------
    print("\n[recommendation] recommend() per query")
    from app.evaluation.recommendations import _query_books, _scenario_env, _syllabus, _book, REQUIRED_TOPICS, SUBTOPIC_MAP

    _, recommender = _scenario_env(
        [
            _syllabus("doc-syllabus"),
            _book("doc-strong", "Complete Database Systems", [(t, SUBTOPIC_MAP[t]) for t in REQUIRED_TOPICS]),
        ]
    )

    def recommend_once():
        _query_books(recommender, "Which book is best for normalization?")

    summary = _bench(recommend_once, repeats=5)
    print(
        f"   mean {summary['mean_ms']:.2f} ms per query, "
        f"median {summary['median_ms']:.2f} ms, p95 {summary['p95_ms']:.2f} ms"
    )

    print(line)
    print("Profile complete. Bottlenecks are optimized only when measured;")
    print("see docs/EVALUATION.md -> Findings for fixes from the evaluation.")
    print(line)


if __name__ == "__main__":
    main()