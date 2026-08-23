"""Hybrid retriever (Phase 6).

Combines dense vector retrieval (Qdrant) with sparse keyword retrieval
(BM25) and fuses the results, improving exact-match queries (subject codes,
regulation numbers, dates, academic years, question numbers, names, numbers
and specific phrases) where vector similarity alone is weak.

Pipeline per query:

    dense = search_dense(query)      # NormalRetriever -> Qdrant cosine
    sparse = search_sparse(query)    # BM25Index -> keyword scores
    fused = fuse_results(dense, sparse)     # Reciprocal Rank Fusion
    ranked = rerank_results(fused)          # score-based refinement
    -> top-K SearchResults (method="hybrid")

Why Reciprocal Rank Fusion (RRF)? Dense cosine scores live roughly in
[-1, 1] while BM25 scores are unbounded and depend on corpus statistics, so
their raw scales are not comparable and must not be added together. RRF only
uses the *rank* of each result in each list (score = sum of 1/(k + rank)),
which is scale-independent, needs no score calibration, and is robust to
outlier scores from either subsystem.

Why a rerank pass? RRF is rank-only, so once a few lists are merged the dense
similarity can act as a fine-grained tiebreaker. The rerank step blends the
RRF score (rank signal) with the min-max normalized dense score (quality
signal) and sorts; BM25 magnitude never leaks into the final score.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field

from app.retrieval.models import SearchResult
from app.retrieval.normal_retriever import NormalRetriever
from app.retrieval.sparse import BM25Index


@dataclass
class FusedResult:
    """A merged entry carrying the rank-based (RRF) and per-method scores."""

    key: tuple
    rrf_score: float = 0.0
    dense_score: float | None = None
    sparse_score: float | None = None
    result: SearchResult | None = None


class HybridRetriever:
    """Dense + sparse retrieval fused with RRF and refined by a rerank pass."""

    def __init__(
        self,
        dense_retriever: NormalRetriever,
        sparse_index: BM25Index,
        top_k: int = 5,
        rrf_k: float = 60.0,
        rrf_weight: float = 0.7,
        dense_weight: float = 0.3,
        pool_factor: int = 2,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.sparse_index = sparse_index
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.rrf_weight = rrf_weight
        self.dense_weight = dense_weight
        self.pool_factor = pool_factor

    # --- Dense (vector) retrieval ----------------------------------------

    async def search_dense(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Vector retrieval over Qdrant via the ``NormalRetriever``."""
        return await self.dense_retriever.retrieve(query, top_k=top_k)

    # --- Sparse (keyword) retrieval --------------------------------------

    def search_sparse(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """BM25 keyword retrieval over the sparse index."""
        hits = self.sparse_index.search(query, top_k=top_k)
        results: list[SearchResult] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                SearchResult(
                    text=payload.get("text", ""),
                    score=hit.score,
                    metadata={key: value for key, value in payload.items() if key != "text"},
                    collection=hit.collection,
                    method="sparse",
                )
            )
        return results

    # --- Fusion (Reciprocal Rank Fusion) ----------------------------------

    def fuse_results(
        self,
        dense: list[SearchResult],
        sparse: list[SearchResult],
        k: float | None = None,
    ) -> list[FusedResult]:
        """Merge dense and sparse result lists using Reciprocal Rank Fusion."""
        k = k or self.rrf_k
        fused: dict[tuple, FusedResult] = {}

        for rank, result in enumerate(dense, start=1):
            key = (result.collection, result.metadata.get("chunk_id"))
            entry = fused.setdefault(key, FusedResult(key=key))
            entry.rrf_score += 1.0 / (k + rank)
            entry.dense_score = result.score
            entry.result = result

        for rank, result in enumerate(sparse, start=1):
            key = (result.collection, result.metadata.get("chunk_id"))
            entry = fused.setdefault(key, FusedResult(key=key))
            entry.rrf_score += 1.0 / (k + rank)
            entry.sparse_score = result.score
            if entry.result is None:
                entry.result = result

        return list(fused.values())

    # --- Reranking --------------------------------------------------------

    def rerank_results(self, fused: list[FusedResult], top_k: int | None = None) -> list[SearchResult]:
        """Blend RRF (rank) with normalized dense similarity and sort."""
        if not fused:
            return []

        max_rrf = max(entry.rrf_score for entry in fused)
        dense_scores = [entry.dense_score for entry in fused if entry.dense_score is not None]
        dense_min = min(dense_scores) if dense_scores else 0.0
        dense_max = max(dense_scores) if dense_scores else 0.0

        def _dense_norm(entry: FusedResult) -> float:
            if entry.dense_score is None or dense_max == dense_min:
                return 0.0
            return (entry.dense_score - dense_min) / (dense_max - dense_min)

        ranked: list[SearchResult] = []
        for entry in fused:
            if entry.result is None:
                continue
            rrf_norm = entry.rrf_score / max_rrf if max_rrf > 0 else 0.0
            final_score = self.rrf_weight * rrf_norm + self.dense_weight * _dense_norm(entry)
            result = copy.copy(entry.result)
            result.score = round(final_score, 6)
            result.method = "hybrid"
            ranked.append(result)

        ranked.sort(key=lambda result: result.score, reverse=True)
        limit = top_k or self.top_k
        return ranked[:limit]

    # --- Orchestration ----------------------------------------------------

    async def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Hybrid retrieval: dense + sparse -> RRF fusion -> rerank."""
        limit = top_k or self.top_k
        if not query.strip() or limit <= 0:
            return []

        pool_size = max(limit * self.pool_factor, 10)
        dense = await self.search_dense(query, top_k=pool_size)
        sparse = await asyncio.to_thread(self.search_sparse, query, pool_size)
        fused = self.fuse_results(dense, sparse)
        return self.rerank_results(fused, top_k=limit)