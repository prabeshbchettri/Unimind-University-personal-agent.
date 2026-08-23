"""Web search endpoints (Phase 11).

``POST /web/search`` exposes the web search provider directly for verification
and for clients that want raw web results. The chat flow routes through the
adaptive router instead.
"""

from fastapi import APIRouter, Request

from app.schemas import WebResultSchema, WebSearchRequest, WebSearchResponse

router = APIRouter(prefix="/web", tags=["web"])


@router.post("/search", response_model=WebSearchResponse, summary="Search the web")
async def web_search(request: Request, body: WebSearchRequest) -> WebSearchResponse:
    provider = request.app.state.web_search_provider
    results = await provider.search(body.query, top_k=body.top_k or 4)
    return WebSearchResponse(
        query=body.query,
        backend=provider.name,
        results=[
            WebResultSchema(
                title=result.title,
                url=result.url,
                snippet=result.snippet,
                retrieved_at=result.retrieved_at,
                score=result.score,
            )
            for result in results
        ],
    )
