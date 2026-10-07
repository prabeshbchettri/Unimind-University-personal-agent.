"""FastAPI application entry point.

Endpoints:

- ``GET /health``: process liveness (Phase 1)
- ``POST /api/chat``: grounded question answering over the ingested corpus
  (Phase 3): retrieve -> build context -> LLM -> answer + citations.

Heavy objects (embedding model, vector store, LLM client, retriever) are
created lazily once per process and shared across requests.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.bm25 import build_bm25_index
from app.config import get_settings
from app.embeddings import EmbeddingError, build_embedding_client
from app.llm import LLMConfigurationError, LLMError, build_llm_client
from app.rag import RAGResult, answer_question, stream_answer
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


class HistoryMessage(BaseModel):
    """One prior turn supplied by the client for conversational context."""

    role: str = Field(pattern=r"^(user|assistant)$")
    content: str = Field(min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    """Request body of the chat endpoint."""

    message: str = Field(min_length=1, max_length=2000)
    # Prior turns, oldest first. Stateless: the client sends the visible
    # conversation back with every request; the server keeps nothing.
    history: list[HistoryMessage] = Field(default_factory=list, max_length=20)


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
        history = [item.model_dump() for item in request.history]
        result: RAGResult = answer_question(
            request.message,
            settings=settings,
            retriever=pipeline["retriever"],
            llm=pipeline["llm"],
            router=pipeline.get("router"),
            hybrid_retriever=pipeline.get("hybrid_retriever"),
            history=history,
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
            "chat request_id=%s strategy=%s strategy_reason=%s retrieval_count=%s provider=%s latency_ms=%d",
            request_id,
            result.strategy if result else "-",
            result.strategy_reason if result else "-",
            len(result.retrieved) if result else "-",
            result.provider if result else "-",
            (time.perf_counter() - started) * 1000,
        )

    return ChatResponse(**result.to_api_dict())


def _sse_error(message: str) -> str:
    """One terminal SSE error event (client-presentable text only)."""
    return f"data: {json.dumps({'type': 'error', 'message': message})}\n\n"


@app.post("/api/chat/stream", tags=["chat"])
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Stream a grounded answer as Server-Sent Events (SSE).

    Same pipeline and guarantees as ``POST /api/chat``; the answer deltas
    stream as they are generated. Event types: ``meta`` (once, before the
    first delta), ``delta`` (repeated), ``done`` (final API-shaped payload),
    ``error`` (terminal). Media type is ``text/event-stream``; Vite's proxy
    and the CORS middleware pass it through unchanged.
    """
    request_id = uuid4().hex[:8]
    if not request.message.strip():
        raise HTTPException(status_code=422, detail="Message must not be empty")

    def event_stream() -> Iterator[str]:
        started = time.perf_counter()
        result: RAGResult | None = None
        history = [item.model_dump() for item in request.history]
        try:
            pipeline = get_pipeline()
            events = stream_answer(
                request.message,
                settings=settings,
                retriever=pipeline["retriever"],
                llm=pipeline["llm"],
                router=pipeline.get("router"),
                hybrid_retriever=pipeline.get("hybrid_retriever"),
                history=history,
            )
            for event in events:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event["type"] == "done":
                    result = RAGResult(
                        answer=event["answer"],
                        sources=[],
                        enough_evidence=event["enough_evidence"],
                        retrieved=[],
                        provider=event["provider"],
                        model=event["model"],
                        strategy=event["strategy"],
                        strategy_reason=event["strategy_reason"],
                    )
        except (RetrievalError, EmbeddingError, VectorStoreError) as exc:
            logger.exception("Retrieval pipeline failure (stream)")
            yield _sse_error("The assistant is temporarily unavailable. Please try again shortly.")
        except LLMConfigurationError as exc:
            logger.exception("Generation configuration error (stream)")
            yield _sse_error("The assistant is misconfigured. Please contact the operator.")
        except LLMError as exc:
            logger.exception("Generation failure (stream)")
            yield _sse_error("The assistant backend reported an error while answering.")
        except ValueError as exc:
            yield _sse_error(str(exc))
        finally:
            if result is not None:
                logger.info(
                    "chat_stream request_id=%s strategy=%s provider=%s latency_ms=%d",
                    request_id,
                    result.strategy,
                    result.provider,
                    (time.perf_counter() - started) * 1000,
                )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )