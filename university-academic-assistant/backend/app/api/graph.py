"""Knowledge-graph endpoints (Phase 7)."""

from fastapi import APIRouter, Request

from app.schemas import GraphSummarySchema

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/summary", response_model=GraphSummarySchema, summary="Knowledge graph summary")
async def graph_summary(request: Request) -> GraphSummarySchema:
    """Return the knowledge-graph status and per-entity counts."""
    return await request.app.state.graph_service.summary()
