"""Retrieval evaluation metrics (Phase 13).

Standard ranking metrics computed at the document level: a retrieved
:class:`SearchResult` is relevant when it belongs to one of the labeled
relevant documents for the query.
"""

from __future__ import annotations

from collections.abc import Sequence


def recall_at_k(relevant_docs: set[str], retrieved_docs: Sequence[str], k: int) -> float:
    """Fraction of relevant documents retrieved within the top ``k``."""
    if not relevant_docs:
        return 0.0
    retrieved = set(retrieved_docs[:k])
    return len(retrieved & relevant_docs) / len(relevant_docs)


def precision_at_k(relevant_docs: set[str], retrieved_docs: Sequence[str], k: int) -> float:
    """Fraction of the top ``k`` results that are relevant."""
    if k <= 0:
        return 0.0
    retrieved = retrieved_docs[:k]
    if not retrieved:
        return 0.0
    return len(set(retrieved) & relevant_docs) / len(retrieved)


def mrr(relevant_docs: set[str], retrieved_docs: Sequence[str]) -> float:
    """Reciprocal rank of the first relevant result (0 when none found)."""
    if not relevant_docs:
        return 0.0
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc in relevant_docs:
            return 1.0 / rank
    return 0.0


def macro_average(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0