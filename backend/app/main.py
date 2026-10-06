"""FastAPI application entry point.

Endpoints:

- ``GET /health``: process liveness (Phase 1)
- ``POST /api/chat``: grounded question answering over the ingested corpus
  (Phase 3): retrieve -> build context -> LLM -> answer + citations.

Heavy objects (embedding model, vector store, LLM client, retriever) are
created lazily once per process and shared across requests.
"""

from __future__ import annotations

import logging
import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.bm25 import build_bm25_index
from app.config import get_settings
from app.embeddings import EmbeddingError, build_embedding_client
from app.llm import LLMConfigurationError, LLMError, build_llm_client
from app.rag import RAGResult, answer_question
from app.retriever import HybridRetriever, RetrievalError, VectorRetriever
from app.routing import QueryRouter
from app.vectorstore import QdrantVectorStore, VectorStoreError

logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(title=settings.app_name)

# The React dev server runs on a different origin, so browser calls need CORS.
_origins = [
    origin.strip()
    for origin in settings.cors_allowed_origins.split(",")
    if origin.strip()
]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )


class HealthResponse(BaseModel):
    """Response body of the health check endpoint."""

    status: str
    service: str
    environment: str


class ReadyResponse(BaseModel):
    """Response body of the readiness endpoint."""

    status: str
    checks: dict[str, str]


class ChatRequest(BaseModel):
    """Request body of the chat endpoint."""

    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    """Response body of the chat endpoint."""

    answer: str
    sources: list[dict]
    enough_evidence: bool
    retrieval: dict
    provider: str
    model: str
    strategy: str
    strategy_reason: str


_pipeline_cache: dict = {}


def get_pipeline() -> dict:
    """Lazily build the shared pipeline objects on first use.

    Module-level so tests can override it with lightweight fakes.
    """
    if not _pipeline_cache:
        embedding = build_embedding_client(settings)
        store = QdrantVectorStore(
            url=settings.qdrant_url,
            collection=settings.qdrant_collection,
            dim=embedding.dim,
        )
        vector_retriever = VectorRetriever(
            store=store, embedding=embedding, top_k=settings.retrieval_top_k
        )
        bm25_index = build_bm25_index(store)
        _pipeline_cache["store"] = store
        _pipeline_cache["retriever"] = vector_retriever
        _pipeline_cache["hybrid_retriever"] = HybridRetriever(
            vector=vector_retriever, bm25=bm25_index, top_k=settings.retrieval_top_k
        )
        _pipeline_cache["llm"] = build_llm_client(settings)
        _pipeline_cache["router"] = QueryRouter()
    return _pipeline_cache


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Report that the API process is running.

    Liveness only: a 200 here does not mean the vector store or the LLM
    providers are usable -- see ``/ready`` for dependency checks.
    """
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        environment=settings.app_env,
    )


@app.get("/ready", response_model=ReadyResponse, tags=["system"])
def ready() -> ReadyResponse:
    """Verify that the retrieval stack actually answers.

    Builds the shared pipeline on first call (embedding model, Qdrant) and
    pings the vector store. Returns 503 with the failing check when a
    dependency is unusable. LLM providers are intentionally not probed: Groq
    would cost an API call per readiness request and Ollama loads the model,
    which is exactly what the timeout/fallback handling is for.
    """
    try:
        pipeline = get_pipeline()
        pipeline["store"].ping()
    except (EmbeddingError, VectorStoreError) as exc:
        logger.warning("Readiness check failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail={"status": "unready", "checks": {"vector_store": str(exc)}},
        ) from exc
    return ReadyResponse(
        status="ready", checks={"vector_store": "ok", "embedding_model": "loaded"}
    )


@app.post("/api/chat", response_model=ChatResponse, tags=["chat"])
def chat(request: ChatRequest) -> ChatResponse:
    """Answer a question grounded in the ingested university documents."""
    request_id = uuid4().hex[:8]
    started = time.perf_counter()
    result: RAGResult | None = None
    try:
        pipeline = get_pipeline()
        result: RAGResult = answer_question(
            request.message,
            settings=settings,
            retriever=pipeline["retriever"],
            llm=pipeline["llm"],
            router=pipeline.get("router"),
            hybrid_retriever=pipeline.get("hybrid_retriever"),
        )
    except (RetrievalError, EmbeddingError, VectorStoreError) as exc:
        # Infrastructure failures: client-visible 503, details in the server log.
        logger.exception("Retrieval pipeline failure")
        raise HTTPException(status_code=503, detail="retrieval pipeline unavailable") from exc
    except LLMConfigurationError as exc:
        # Deployment problem (missing/bad provider config): not a fallback case.
        logger.exception("Generation configuration error")
        raise HTTPException(status_code=500, detail="generation pipeline misconfigured") from exc
    except LLMError as exc:
        logger.exception("Generation failure")
        raise HTTPException(status_code=503, detail="generation pipeline unavailable") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        # Structured request log with operational metadata only. The question
        # text is deliberately excluded: chat content is sensitive user data.
        logger.info(
            "chat request_id=%s strategy=%s retrieval_count=%s provider=%s latency_ms=%d",
            request_id,
            result.strategy if result else "-",
            len(result.retrieved) if result else "-",
            result.provider if result else "-",
            (time.perf_counter() - started) * 1000,
        )

    return ChatResponse(**result.to_api_dict())
