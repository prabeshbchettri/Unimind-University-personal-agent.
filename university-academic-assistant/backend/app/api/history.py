"""Chat history endpoints (Phase 10).

Session/message persistence is implemented in Phase 10; this endpoint lists
the persisted chat sessions (most recently updated first). Per-session
detail, deletion and message persistence live under ``/chat``.
"""

from fastapi import APIRouter, Query, Request

from app.schemas import ChatSessionSummary, HistoryListResponse

router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=HistoryListResponse, summary="List chat sessions")
async def list_history(
    request: Request,
    limit: int = Query(50, ge=1, le=200, description="Maximum number of sessions."),
    offset: int = Query(0, ge=0, description="Number of sessions to skip."),
) -> HistoryListResponse:
    history = request.app.state.chat_history
    sessions, total = await history.list_sessions(limit=limit, offset=offset)
    return HistoryListResponse(
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