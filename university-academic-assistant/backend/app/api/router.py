"""API router aggregation.

Every versioned endpoint group is mounted here and included in the app.
"""

from fastapi import APIRouter

from . import chat, documents, graph, health, history, recommendations, routing, search, web

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(chat.router)
api_router.include_router(documents.router)
api_router.include_router(graph.router)
api_router.include_router(routing.router)
api_router.include_router(recommendations.router)
api_router.include_router(history.router)
api_router.include_router(search.router)
api_router.include_router(web.router)

__all__ = ["api_router"]