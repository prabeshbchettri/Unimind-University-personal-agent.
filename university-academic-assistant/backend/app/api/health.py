"""Health check endpoint."""

from fastapi import APIRouter, Request

from app.schemas import HealthResponse
from app.services import ChatService, DocumentIngestionService, QueryRouter, RecommendationService

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness check")
async def health(request: Request) -> HealthResponse:
    """Return application liveness status."""
    services: list = [
        request.app.state.ingestion_service,
        request.app.state.query_router,
        request.app.state.recommendation_service,
        request.app.state.chat_service,
        request.app.state.graph_service,
    ]
    # Services report "ok" without contacting any backing store.
    for service in services:
        await service.health()
    return HealthResponse(status="ok")