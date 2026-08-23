"""FastAPI application factory.

``create_app`` builds a fully configured application. This factory pattern
keeps the app testable (tests can build a fresh instance) while allowing a
single long-lived instance for normal serving.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.caching import RetrievalCache
from app.config import settings
from app.core.context import set_request_id
from app.core.logging import get_logger, setup_logging
from app.core.rate_limit import SlidingWindowRateLimiter
from app.embedding import build_embedder
from app.graph.extractor import GraphExtractor
from app.llm import build_llm
from app.rag.context import ContextBuilder
from app.repositories.chat_history import ChatHistoryRepository
from app.repositories.graph import build_graph_repository
from app.repositories.qdrant import QdrantRepository
from app.retrieval.cached import CachedRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.normal_retriever import NormalRetriever
from app.retrieval.sparse import BM25Index
from app.retrieval.web_retriever import WebRetriever
from app.services import ChatService, DocumentIngestionService, QueryRouter, RecommendationService
from app.services.graph import KnowledgeGraphService
from app.services.vector_indexing import VectorIndexingService
from app.web import build_web_search_provider

logger = get_logger("app.main")

# Endpoints exempt from rate limiting: liveness and interactive API docs.
_UNLIMITED_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}


def _client_ip(request: Request) -> str:
    """Client address, honouring the first ``X-Forwarded-For`` hop."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown lifecycle.

    Phase 1 instantiates placeholder services. Phase 4 adds the Qdrant
    repository, the embedder and the vector indexing service. PostgreSQL and
    Neo4j connections arrive in later phases.
    """
    logger.info("Application starting", extra={"extra_fields": {"app": settings.app_name, "version": settings.app_version}})
    app.state.ingestion_service = DocumentIngestionService()
    app.state.query_router = QueryRouter()

    qdrant = QdrantRepository(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_names=settings.collection_names,
        default_dimensions=settings.embedding_dimensions,
    )
    app.state.qdrant_repository = qdrant
    app.state.embedder = build_embedder(settings)
    app.state.sparse_index = BM25Index()

    graph_repository = build_graph_repository(settings)
    app.state.graph_repository = graph_repository
    app.state.graph_service = KnowledgeGraphService(
        repository=graph_repository,
        extractor=GraphExtractor(),
    )

    # Phase 13: process-local retrieval cache, invalidated by any ingestion.
    retrieval_cache = (
        RetrievalCache(
            max_entries=settings.retrieval_cache_max_entries,
            ttl_seconds=settings.retrieval_cache_ttl_seconds,
        )
        if settings.retrieval_cache_enabled
        else None
    )
    app.state.retrieval_cache = retrieval_cache

    app.state.vector_indexing_service = VectorIndexingService(
        embedder=app.state.embedder,
        repository=qdrant,
        collection_names=settings.collection_names,
        sparse_index=app.state.sparse_index,
        graph_service=app.state.graph_service,
        retrieval_cache=retrieval_cache,
    )

    normal_retriever = NormalRetriever(
        embedder=app.state.embedder,
        repository=qdrant,
        collection_names=settings.collection_names,
        top_k=settings.rag_top_k,
        min_score=settings.rag_min_score,
    )
    app.state.normal_retriever = normal_retriever
    hybrid_retriever = HybridRetriever(
        dense_retriever=normal_retriever,
        sparse_index=app.state.sparse_index,
        top_k=settings.rag_top_k,
        rrf_k=settings.hybrid_rrf_k,
        rrf_weight=settings.hybrid_rerank_rrf_weight,
        dense_weight=settings.hybrid_rerank_dense_weight,
        pool_factor=settings.hybrid_pool_factor,
    )
    app.state.hybrid_retriever = hybrid_retriever
    app.state.context_builder = ContextBuilder(
        max_chars=settings.context_max_chars,
        max_sources=settings.context_max_sources,
    )
    app.state.llm_client = build_llm(settings)

    # Phase 10: persistent chat sessions/messages in PostgreSQL (or SQLite).
    history_repository = ChatHistoryRepository(
        database_url=settings.database_url,
        backend=settings.database_backend,
    )
    app.state.chat_history = history_repository

    # Phase 8: adaptive router decides NORMAL / HYBRID / GRAPH / WEB per query.
    # A fixed strategy ("normal" / "hybrid") keeps the legacy behavior.
    from app.retrieval.graph_retriever import GraphRetriever
    from app.router.analyzers import LLMQueryAnalyzer, RuleBasedQueryAnalyzer
    from app.router.router import AdaptiveRouter

    graph_retriever = GraphRetriever(
        graph_service=app.state.graph_service,
        analyzer=RuleBasedQueryAnalyzer(),
        top_k=settings.rag_top_k,
        dense_retriever=normal_retriever,
    )
    app.state.graph_retriever = graph_retriever

    # Phase 11: controlled web search for current/external questions.
    web_provider = build_web_search_provider(settings)
    app.state.web_search_provider = web_provider
    web_retriever = WebRetriever(
        provider=web_provider,
        top_k=settings.web_max_sources,
        timeout=settings.web_search_timeout,
    )
    app.state.web_retriever = web_retriever

    llm_analyzer = None
    if settings.router_classifier in {"auto", "llm"}:
        llm_analyzer = LLMQueryAnalyzer(app.state.llm_client)

    strategy_mode = settings.rag_retrieval_strategy.lower()
    fixed_strategy = None if strategy_mode == "auto" else strategy_mode
    retriever = AdaptiveRouter(
        normal_retriever=normal_retriever,
        hybrid_retriever=hybrid_retriever,
        graph_retriever=graph_retriever,
        analyzer=RuleBasedQueryAnalyzer(),
        llm_analyzer=llm_analyzer,
        top_k=settings.rag_top_k,
        fixed_strategy=fixed_strategy,
        fallback_strategy=settings.router_fallback_strategy,
        classifier_mode=settings.router_classifier,
        web_retriever=None if fixed_strategy else web_retriever,
    )
    app.state.adaptive_router = retriever
    app.state.retriever = retriever

    # Phase 9: evidence-based library book recommendation.
    app.state.recommendation_service = RecommendationService(
        repository=qdrant,
        embedder=app.state.embedder,
        graph_service=app.state.graph_service,
        llm=app.state.llm_client,
        semantic_threshold=settings.recommendation_semantic_threshold,
        topic_weight=settings.recommendation_topic_weight,
        subtopic_weight=settings.recommendation_subtopic_weight,
    )
    app.state.chat_service = ChatService(
        retriever=CachedRetriever(retriever, retrieval_cache),
        context_builder=app.state.context_builder,
        llm=app.state.llm_client,
        top_k=settings.rag_top_k,
        history=history_repository,
        history_max_chars=settings.history_max_chars,
        history_max_messages=settings.history_max_messages,
    )
    logger.info(
        "Vector index ready",
        extra={"extra_fields": {"embedder": type(app.state.embedder).__name__, "collections": settings.collection_names}},
    )
    logger.info(
        "Retrieval strategy ready",
        extra={
            "extra_fields": {
                "strategy": type(retriever).__name__,
                "mode": fixed_strategy or "auto",
                "sparse_index_size": app.state.sparse_index.size,
            }
        },
    )
    logger.info(
        "Knowledge graph ready",
        extra={"extra_fields": {"backend": type(graph_repository).__name__}},
    )
    logger.info(
        "Chat history ready",
        extra={"extra_fields": {"backend": history_repository.backend, "engine": str(history_repository.engine.url)}},
    )
    logger.info(
        "Web search ready",
        extra={"extra_fields": {"backend": web_provider.name, "timeout_seconds": settings.web_search_timeout}},
    )
    yield
    await app.state.embedder.close()
    await qdrant.close()
    await app.state.llm_client.close()
    await graph_repository.close()
    await history_repository.close()
    logger.info("Application shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    setup_logging()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
    )

    # Phase 12: allow the React frontend (dev server by default) to call the API.
    allowed_origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    # Phase 13: per-IP sliding-window rate limiting (outermost middleware so
    # throttled requests never reach the request logger or the handlers).
    rate_limiter = SlidingWindowRateLimiter(settings.rate_limit_per_minute)

    @app.middleware("http")
    async def rate_limit(request: Request, call_next) -> Response:
        """Reject requests exceeding the per-IP window with HTTP 429."""
        if not settings.rate_limit_enabled or request.url.path in _UNLIMITED_PATHS:
            return await call_next(request)
        client_ip = _client_ip(request)
        if not rate_limiter.allow(client_ip):
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please slow down and try again."},
                headers={"Retry-After": str(rate_limiter.retry_after_seconds(client_ip))},
            )
        return await call_next(request)

    @app.middleware("http")
    async def log_requests(request: Request, call_next) -> Response:
        """Log every API request with a correlation id and latency.

        The id is exposed to the client via ``X-Request-ID`` and injected into
        every log record emitted while handling the request. Only routing
        metadata is logged (method/path/status/latency) — never request bodies,
        user messages, document contents or retrieved text.
        """
        request_id = uuid.uuid4().hex[:12]
        set_request_id(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "HTTP request failed",
                extra={
                    "extra_fields": {
                        "method": request.method,
                        "path": request.url.path,
                        "status": 500,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    }
                },
            )
            raise
        duration_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "HTTP request",
            extra={
                "extra_fields": {
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                }
            },
        )
        return response

    return app


app = create_app()
