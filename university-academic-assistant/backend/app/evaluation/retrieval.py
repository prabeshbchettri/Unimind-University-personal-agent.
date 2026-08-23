"""Retrieval evaluation (Phase 13).

Runs the labeled corpus queries through three retrieval strategies —
Normal (dense), Hybrid (dense + BM25) and Graph — and reports document-level
Recall@K, Precision@K and MRR for K in (1, 3, 5).

Methodology (documented in docs/EVALUATION.md):

- Each strategy is scored only on the queries labeled for it, because the
  graph retriever is *specialized*: it legitimately returns no rows for
  semantic queries (the chat pipeline falls back to dense retrieval for
  those). Averaging graph metrics over all queries would penalize the
  strategy for correctly staying silent.
- ``empty_rate_on_other`` guards the opposite failure: a strategy should not
  produce rows for queries labeled for a different strategy. The graph
  retriever must score 1.0 here (never pollutes); normal/hybrid scoring high
  here is expected and informative, not an error.
"""

from __future__ import annotations

import asyncio

from app.evaluation.corpus import RETRIEVAL_QUERIES
from app.evaluation.environment import EvaluationEnvironment, document_id_of
from app.evaluation.metrics import macro_average, mrr, precision_at_k, recall_at_k
from app.retrieval.graph_retriever import GraphRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.normal_retriever import NormalRetriever


def _run(coro):
    return asyncio.run(coro)


def _retrieve_docs(retriever, query: str, top_k: int) -> list[str]:
    results = _run(retriever.retrieve(query, top_k=top_k))
    return [document_id_of(result) for result in results]


def evaluate_retrieval(env: EvaluationEnvironment, top_k: int = 5) -> dict:
    """Evaluate all three strategies on the labeled queries."""
    strategies = {
        "NORMAL": env.normal_retriever,
        "HYBRID": env.hybrid_retriever,
        "GRAPH": env.graph_retriever,
    }
    output: dict = {"queries": [], "strategies": {}, "top_k": top_k}

    for name, retriever in strategies.items():
        labeled = [item for item in RETRIEVAL_QUERIES if item["expected_strategy"] == name]
        other = [item for item in RETRIEVAL_QUERIES if item["expected_strategy"] != name]

        recall_1, recall_3, recall_5 = [], [], []
        precision_3, precision_5 = [], []
        mrr_values = []
        rows = []
        for item in labeled:
            relevant = set(item["expected_docs"])
            retrieved = _retrieve_docs(retriever, item["query"], top_k)
            row = {
                "query": item["query"],
                "category": item["category"],
                "relevant": sorted(relevant),
                "retrieved": retrieved,
                "recall@5": recall_at_k(relevant, retrieved, 5),
                "mrr": mrr(relevant, retrieved),
            }
            rows.append(row)
            recall_1.append(recall_at_k(relevant, retrieved, 1))
            recall_3.append(recall_at_k(relevant, retrieved, 3))
            recall_5.append(recall_at_k(relevant, retrieved, 5))
            precision_3.append(precision_at_k(relevant, retrieved, 3))
            precision_5.append(precision_at_k(relevant, retrieved, 5))
            mrr_values.append(row["mrr"])

        empty_on_other = sum(
            not _retrieve_docs(retriever, item["query"], top_k) for item in other
        ) / len(other)

        output["strategies"][name] = {
            "label": name,
            "evaluated_queries": len(labeled),
            "recall@1": macro_average(recall_1),
            "recall@3": macro_average(recall_3),
            "recall@5": macro_average(recall_5),
            "precision@3": macro_average(precision_3),
            "precision@5": macro_average(precision_5),
            "mrr": macro_average(mrr_values),
            "empty_rate_on_other_queries": empty_on_other,
            "rows": rows,
        }
    output["queries"] = [
        {
            "query": item["query"],
            "category": item["category"],
            "expected_strategy": item["expected_strategy"],
            "expected_docs": sorted(item["expected_docs"]),
        }
        for item in RETRIEVAL_QUERIES
    ]
    return output