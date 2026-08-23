"""Vector search endpoints (Phase 4).

Basic semantic search over the indexed Qdrant collections. RAG answer
generation builds on this in Phase 5.
"""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from app.schemas import SearchResponseSchema, SearchResultSchema

router = APIRouter(prefix="/search", tags=["search"])


@router.get("", response_model=SearchResponseSchema, summary="Semantic search over indexed documents")
async def search(
    request: Request,
    q: str = Query(..., min_length=1, description="Search query"),
    collection: str | None = Query(None, description="Collection to search (default: all)"),
    top_k: int = Query(5, ge=1, le=50, description="Number of results"),
) -> SearchResponseSchema:
    """Return the most relevant indexed chunks for ``q``.

    Each result carries the chunk ``text``, its similarity ``score`` and the
    chunk ``metadata`` (document id/type, page, topic, subject, ...).
    """
    results = await request.app.state.vector_indexing_service.search(
        q,
        collection=collection,
        top_k=top_k,
    )
    return SearchResponseSchema(
        query=q,
        collection=collection,
        top_k=top_k,
        results=[
            SearchResultSchema(text=result.text, score=result.score, metadata=result.metadata)
            for result in results
        ],
    )