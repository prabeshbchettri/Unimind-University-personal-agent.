"""Chat endpoints (Phase 5 + Phase 10).

``POST /chat`` answers a message grounded on retrieved university context.
Phase 10 adds conversation persistence: the request may carry a
``session_id`` (continue an existing conversation) or omit it (start a new
one); the user message and the assistant answer are stored, and the bounded
recent history is included in the generation prompt. Session management is
exposed under ``/chat/sessions``.
"""

from fastapi import APIRouter, HTTPException, Query, Request, Response

from app.embedding.errors import EmbeddingUnavailableError
from app.llm.errors import LLMUnavailableError
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ChatSessionDetail,
    ChatMessageSchema,
    ChatSessionSummary,
    SessionListResponse,
    SourceSchema,
)
from app.services.chat import SessionNotFoundError

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse, summary="Send a chat message")
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    chat_service = request.app.state.chat_service
    try:
        result = await chat_service.answer(body.message, top_k=body.top_k, session_id=body.session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (LLMUnavailableError, EmbeddingUnavailableError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    router = request.app.state.adaptive_router
    strategy = "NORMAL"
    if router is not None and getattr(router, "last_plan", None) is not None:
        strategy = router.last_plan.strategy.value

    return ChatResponse(
        answer=result.answer,
        session_id=result.session_id,
        strategy=strategy,
        sources=[
            SourceSchema(
                text=source.text,
                score=source.score,
                metadata=source.metadata,
                collection=source.collection or "",
                retrieval_method=source.method,
                kind="web" if source.collection == "web" else "university",
                url=source.metadata.get("url", "") if source.collection == "web" else "",
                retrieved_at=source.metadata.get("retrieved_at", "") if source.collection == "web" else "",
            )
            for source in result.sources
        ],
    )


@router.get("/sessions", response_model=SessionListResponse, summary="List chat sessions")
async def list_sessions(
    request: Request,
    limit: int = Query(50, ge=1, le=200, description="Maximum number of sessions."),
    offset: int = Query(0, ge=0, description="Number of sessions to skip."),
) -> SessionListResponse:
    history = request.app.state.chat_history
    sessions, total = await history.list_sessions(limit=limit, offset=offset)
    return SessionListResponse(
        items=[
            ChatSessionSummary(
                session_id=session.session_id,
                title=session.title,
                created_at=session.created_at,
                updated_at=session.updated_at,
                message_count=count,
            )
            for session, count in sessions
        ],
        total=total,
    )


@router.get("/sessions/{session_id}", response_model=ChatSessionDetail, summary="Get a chat session")
async def get_session(request: Request, session_id: str) -> ChatSessionDetail:
    history = request.app.state.chat_history
    session = await history.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    messages = await history.get_messages(session_id)
    return ChatSessionDetail(
        session_id=session.session_id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        message_count=len(messages),
        messages=[
            ChatMessageSchema(
                message_id=message.message_id,
                session_id=message.session_id,
                role=message.role,
                content=message.content,
                created_at=message.created_at,
            )
            for message in messages
        ],
    )


@router.delete("/sessions/{session_id}", status_code=204, summary="Delete a chat session")
async def delete_session(request: Request, session_id: str) -> Response:
    history = request.app.state.chat_history
    deleted = await history.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return Response(status_code=204)