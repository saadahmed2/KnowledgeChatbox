"""
/chat  endpoints — synchronous and Server-Sent Event streaming.
"""
from __future__ import annotations

import asyncio
from typing import Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.api.schemas import ChatRequest, ChatResponse, ResetResponse, SourceDocument
from app.chains.rag_chain import ChatSession
from app.retrieval.vector_store import VectorStoreManager
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])

# ── Per-session state (in-process; swap for Redis in production) ──────────────
_vs_manager = VectorStoreManager()
_sessions: Dict[str, ChatSession] = {}


def _get_session(session_id: str) -> ChatSession:
    if session_id not in _sessions:
        _sessions[session_id] = ChatSession(_vs_manager)
        logger.info("New chat session created: %s", session_id)
    return _sessions[session_id]


def _format_sources(raw_sources: list) -> list[SourceDocument]:
    seen: set[str] = set()
    result = []
    for doc in raw_sources:
        key = f"{doc.metadata.get('source')}:{doc.metadata.get('chunk_index')}"
        if key in seen:
            continue
        seen.add(key)
        result.append(
            SourceDocument(
                content=doc.page_content[:500],
                source=doc.metadata.get("source", ""),
                filename=doc.metadata.get("filename", ""),
                page=doc.metadata.get("page", None),
                chunk_index=doc.metadata.get("chunk_index", None),
            )
        )
    return result


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/", response_model=ChatResponse, summary="Ask a question")
async def chat(request: ChatRequest):
    """
    Synchronous Q&A against the ingested knowledge base.
    For streaming responses, set `stream: true` and use GET /chat/stream.
    """
    session = _get_session(request.session_id)
    try:
        answer, sources = await session.achat(request.question)
    except Exception as exc:
        logger.error("Chat error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(
        answer=answer,
        session_id=request.session_id,
        sources=_format_sources(sources),
    )


@router.post("/stream", summary="Streaming Q&A (SSE)")
async def chat_stream(request: ChatRequest):
    """
    Token-level streaming via Server-Sent Events.
    Each event is a plain text chunk; the stream ends with `[DONE]`.
    """
    session = _get_session(request.session_id)

    async def event_generator():
        try:
            async for token in session.astream(request.question):
                yield f"data: {token}\n\n"
        except Exception as exc:
            logger.error("Streaming error: %s", exc, exc_info=True)
            yield f"data: [ERROR] {exc}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.delete("/{session_id}/history", response_model=ResetResponse)
async def clear_history(session_id: str):
    """Clear conversation history for the given session."""
    if session_id in _sessions:
        _sessions[session_id].reset_history()
    return ResetResponse(message=f"History cleared for session '{session_id}'.")


@router.delete("/{session_id}", response_model=ResetResponse)
async def delete_session(session_id: str):
    """Destroy the session entirely."""
    _sessions.pop(session_id, None)
    return ResetResponse(message=f"Session '{session_id}' deleted.")
