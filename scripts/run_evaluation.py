"""Run the manual evaluation dataset through the real RAG pipeline.

Executed in-process (same embedding model, vector store, router, LLM chain as
the API server) so the numbers describe the production code path, not a test
double. This is a manual-engineering evaluation on a tiny synthetic corpus --
the output is explicitly labeled as such and is NOT a scientific benchmark.

Usage (from backend/):

    python ../scripts/run_evaluation.py [--json out.json]

Note: the Qdrant embedded engine takes an exclusive file lock, so the API
server must not run at the same time.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

# Windows consoles default to cp1252; model output can contain any Unicode.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from app.config import get_settings  # noqa: E402
from app.rag import INSUFFICIENT_EVIDENCE_ANSWER, answer_question  # noqa: E402

DATASET = Path(__file__).resolve().parent.parent / "backend" / "evaluation" / "dataset.json"

_CITATION_RE = re.compile(r"\[\d+\]")


def build_pipeline(settings):
    """Build the same shared objects the API server uses."""
    from app.bm25 import build_bm25_index
    from app.embeddings import build_embedding_client
    from app.llm import build_llm_client
    from app.retriever import HybridRetriever, VectorRetriever
    from app.routing import QueryRouter
    from app.vectorstore import QdrantVectorStore

    embedding = build_embedding_client(settings)
    store = QdrantVectorStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        dim=embedding.dim,
    )
    vector_retriever = VectorRetriever(
        store=store, embedding=embedding, top_k=settings.retrieval_top_k
    )
    return {
        "retriever": vector_retriever,
        "hybrid_retriever": HybridRetriever(
            vector=vector_retriever, bm25=build_bm25_index(store), top_k=settings.retrieval_top_k
        ),
        "llm": build_llm_client(settings),
        "router": QueryRouter(),
    }


def run_case(case: dict, pipeline: dict, settings) -> dict:
    """Execute one dataset case and record measured signals."""
    question = case["question"]
    started = time.perf_counter()
    result = answer_question(
        question,
        settings=settings,
        retriever=pipeline["retriever"],
        llm=pipeline["llm"],
        router=pipeline["router"],
        hybrid_retriever=pipeline["hybrid_retriever"],
    )
    latency_ms = (time.perf_counter() - started) * 1000

    docs = [source.document for source in result.sources]
    expected_doc = case.get("relevant_source")

    # Retrieval relevance: is an expected document present in the cited sources?
    top1_relevant = bool(docs) and (expected_doc is None or docs[0] == expected_doc)
    top3_relevant = expected_doc is None or expected_doc in docs[:3]

    # Routing correctness: did the router pick the expected strategy?
    routing_ok = result.strategy == case["expected_strategy"]

    # Answer checks.
    answer_lower = result.answer.lower()
    if "answer_must_mention" in case:
        mention_ok = all(term.lower() in answer_lower for term in case["answer_must_mention"])
    elif case.get("answer_must_mention_any"):
        mention_ok = any(
            term.lower() in answer_lower for term in case["answer_must_mention_any"]
        )
    else:
        mention_ok = None  # decline-case: no content expectation beyond the gate

    evidence_ok = result.enough_evidence is not case["insufficient_evidence_expected"]

    # Citation presence: only meaningful for grounded answers.
    citations_present = (
        bool(_CITATION_RE.search(result.answer)) if result.enough_evidence else None
    )

    # Groundedness gate: an expected-decline case must return the controlled text.
    decline_ok = (
        (result.answer == INSUFFICIENT_EVIDENCE_ANSWER)
        if case["insufficient_evidence_expected"]
        else None
    )

    return {
        "id": case["id"],
        "category": case["category"],
        "question": question,
        "answer": result.answer,
        "provider": result.provider,
        "model": result.model,
        "strategy": result.strategy,
        "expected_strategy": case["expected_strategy"],
        "routing_ok": routing_ok,
        "insufficient_evidence": not result.enough_evidence,
        "evidence_ok": evidence_ok,
        "citations_present": citations_present,
        "cited_documents": docs,
        "top1_relevant": top1_relevant,
        "top3_relevant": top3_relevant,
        "mention_ok": mention_ok,
        "decline_ok": decline_ok,
        "chunks_retrieved": len(result.retrieved),
        "top_score": round(
            max(
                (
                    chunk.semantic_score
                    for chunk in result.retrieved
                    if chunk.semantic_score is not None
                ),
                default=0.0,
            ),
            3,
        ),
        "latency_ms": round(latency_ms),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", dest="json_out", default=None, help="optional JSON output path")
    args = parser.parse_args()

    get_settings.cache_clear()
    settings = get_settings()
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    cases = dataset["cases"]

    print(f"Building pipeline (provider={settings.llm_provider}, store={settings.qdrant_url})...")
    pipeline = build_pipeline(settings)

    results = []
    failures = 0
    for case in cases:
        row = run_case(case, pipeline, settings)
        # A case passes when evidence handling, routing, content and citations agree.
        checks = [row["evidence_ok"], row["routing_ok"], row["top3_relevant"]]
        checks.append(row["decline_ok"] if row["decline_ok"] is not None else True)
        checks.append(row["mention_ok"] if row["mention_ok"] is not None else True)
        checks.append(row["citations_present"] if row["citations_present"] is not None else True)
        row["pass"] = all(checks)
        failures += 0 if row["pass"] else 1
        flag = "PASS" if row["pass"] else "FAIL"
        print(
            f"[{flag}] {row['id']:>10}  strategy={row['strategy']:<6}"
            f"(exp {row['expected_strategy']:<6}) provider={row['provider']:<16}"
            f" top_score={row['top_score']:<6} {row['latency_ms']:>5} ms"
        )
        if not row["pass"]:
            print(f"         answer: {row['answer'][:140]}")
            print(f"         cited:  {row['cited_documents']}")
        results.append(row)

    total = len(results)
    strategies: dict[str, int] = {}
    for row in results:
        strategies[row["strategy"]] = strategies.get(row["strategy"], 0) + 1
    routing_correct = sum(1 for row in results if row["routing_ok"])
    grounded = sum(1 for row in results if row["evidence_ok"])
    cited = sum(
        1 for row in results if row["citations_present"] is not None and row["citations_present"]
    )
    declines = [row for row in results if row["insufficient_evidence"]]
    latencies = sorted(row["latency_ms"] for row in results)

    print("\n=== SUMMARY (manual engineering evaluation, not a benchmark) ===")
    print(f"cases: {total} | pass: {total - failures} | fail: {failures}")
    print(f"routing matched expectation: {routing_correct}/{total}")
    print(f"evidence gate behaved as expected: {grounded}/{total}")
    print(f"grounded answers with citations: {cited}")
    print(f"controlled declines: {len(declines)} -> {[row['id'] for row in declines]}")
    print(f"strategy distribution: {strategies}")
    print(
        "latency ms  min/median/max: "
        f"{latencies[0]}/{latencies[len(latencies) // 2]}/{latencies[-1]}"
    )

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"results": results}, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"JSON results written to {args.json_out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
