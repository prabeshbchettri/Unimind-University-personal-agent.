"""Pydantic schemas shared across the API."""

from .chat import ChatRequest, ChatResponse, SourceSchema
from .chat_history import (
    ChatMessageSchema,
    ChatSessionDetail,
    ChatSessionSummary,
    SessionListResponse,
)
from .documents import (
    DocumentListResponse,
    DocumentResponse,
    DocumentUploadResponse,
    NormalizedDocumentSchema,
    PageContentSchema,
)
from .graph import GraphSummarySchema
from .health import HealthResponse
from .history import HistoryListResponse, HistoryResponse
from .recommendations import (
    BookRecommendationSchema,
    RecommendationRequest,
    RecommendationResponse,
)
from .routing import QueryPlanRequest, RetrievalPlanSchema
from .search import IndexResultSchema, SearchResponseSchema, SearchResultSchema
from .web import WebResultSchema, WebSearchRequest, WebSearchResponse

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "SourceSchema",
    "ChatMessageSchema",
    "ChatSessionDetail",
    "ChatSessionSummary",
    "SessionListResponse",
    "DocumentResponse",
    "DocumentListResponse",
    "DocumentUploadResponse",
    "NormalizedDocumentSchema",
    "PageContentSchema",
    "GraphSummarySchema",
    "HealthResponse",
    "HistoryResponse",
    "HistoryListResponse",
    "BookRecommendationSchema",
    "RecommendationRequest",
    "RecommendationResponse",
    "QueryPlanRequest",
    "RetrievalPlanSchema",
    "IndexResultSchema",
    "SearchResponseSchema",
    "SearchResultSchema",
    "WebResultSchema",
    "WebSearchRequest",
    "WebSearchResponse",
]