"""Retrieval strategies: query -> ranked evidence chunks with metadata.

The read-side of the pipeline. ``answer_question`` feeds the query to the
strategy the router selected, so each strategy returns the same shape:

- :class:`VectorRetriever` (strategy ``normal``) -- semantic/vector similarity
  only, unchanged from the Phase 3 pipeline.
- :class:`HybridRetriever` (strategy ``hybrid``) -- vector plus BM25 candidates
  fused with Reciprocal Rank Fusion (RRF), deduplicated by chunk id, with per
  strategy scores preserved so the context builder can still apply relevance
  gates.

RRF ranks by *position* rather than raw score, so the two strategies never have
to produce comparable score scales; the result lists are not simply
concatenated. Source metadata (chunk id, document, page) survives fusion and is
returned on every chunk.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.bm25 import BM25Index
from app.embeddings import EmbeddingClient
from app.vectorstore import QdrantVectorStore


class RetrievalError(RuntimeError):
    """Raised when retrieval cannot be performed."""


@dataclass(frozen=True)
class RetrievedChunk:
    """One ranked evidence chunk returned for a query.

    ``score`` is the display score: the semantic cosine similarity when the
    chunk was retrieved by the vector search, otherwise its BM25 score.
    ``semantic_score``/``bm25_score`` keep the per-strategy values so the
    context builder can apply the correct relevance gate.
    """

    chunk_id: str
    text: str
    document: str
    page: int
    score: float
    chunk_index: int
    semantic_score: float | None = None
    bm25_score: float | None = None


class VectorRetriever:
    """Retrieve the most similar chunks for a query from the vector store."""

    def __init__(self, store: QdrantVectorStore, embedding: EmbeddingClient, top_k: int) -> None:
        if top_k <= 0:
            raise RetrievalError(f"top_k must be positive, got {top_k}")
        self._store = store
        self._embedding = embedding
        self._top_k = top_k

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Embed the query and return ranked chunks, best first."""
        query = query.strip()
        if not query:
            raise RetrievalError("Query must not be empty")
        query_vector = self._embedding.embed_query(query)
        results = self._store.search(query_vector=query_vector, top_k=self._top_k)
        return [
            RetrievedChunk(
                chunk_id=result.chunk_id,
                text=result.text,
                document=result.document_name,
                page=result.page,
                score=result.score,
                chunk_index=result.chunk_index,
                semantic_score=result.score,
            )
            for result in results
        ]


def _rrf_combine(ranked_ids: list[list[str]], *, k: int = 60) -> list[str]:
    """Fuse ranked id lists with Reciprocal Rank Fusion, in fused rank order.

    Each item in ``ranked_ids`` is a list of chunk ids ranked best-first.
    A chunk present in several lists appears once; its fused score is the sum
    of ``1 / (k + rank)``. With ``k=60`` the combined ranking is dominated by
    *position* (not raw score), which makes scores from different strategies
    directly comparable.
    """
    fused: dict[str, float] = {}
    for ranked in ranked_ids:
        for rank, chunk_id in enumerate(ranked):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return [chunk_id for chunk_id, _ in sorted(fused.items(), key=lambda item: item[1], reverse=True)]


class HybridRetriever:
    """Vector + BM25 retrieval fused by RRF, deduplicated, with scores kept."""

    name = "hybrid"

    def __init__(self, vector: VectorRetriever, bm25: BM25Index, top_k: int) -> None:
        if top_k <= 0:
            raise RetrievalError(f"top_k must be positive, got {top_k}")
        self._vector = vector
        self._bm25 = bm25
        self._top_k = top_k

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Return at most ``top_k`` chunks fused from both strategies."""
        query = query.strip()
        if not query:
            raise RetrievalError("Query must not be empty")

        vector_results = self._vector.retrieve(query)
        lexical_hits = self._bm25.search(query, top_k=self._top_k)

        ordered_ids = _rrf_combine(
            [
                [chunk.chunk_id for chunk in vector_results],
                [hit.chunk_id for hit in lexical_hits],
            ]
        )

        vector_by_id = {chunk.chunk_id: chunk for chunk in vector_results}
        lexical_by_id = {hit.chunk_id: hit for hit in lexical_hits}

        combined: list[RetrievedChunk] = []
        for chunk_id in ordered_ids:
            vector_item = vector_by_id.get(chunk_id)
            lexical_item = lexical_by_id.get(chunk_id)
            # At least one strategy returned the chunk (both are keyed by id).
            source = vector_item if vector_item is not None else lexical_item
            semantic_score = vector_item.score if vector_item is not None else None
            bm25_score = lexical_item.score if lexical_item is not None else None
            combined.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=source.text,
                    document=source.document,
                    page=source.page,
                    score=semantic_score if semantic_score is not None else bm25_score,
                    chunk_index=source.chunk_index,
                    semantic_score=semantic_score,
                    bm25_score=bm25_score,
                )
            )
            if len(combined) >= self._top_k:
                break
        return combined
