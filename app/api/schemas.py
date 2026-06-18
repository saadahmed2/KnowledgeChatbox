"""
Pydantic request / response schemas shared across API routes.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4096, description="User question")
    session_id: str = Field(default="default", description="Session identifier for multi-user support")
    stream: bool = Field(default=False, description="Enable token-level streaming (SSE)")


class SourceDocument(BaseModel):
    content: str
    source: str
    filename: str
    page: Optional[int] = None
    chunk_index: Optional[int] = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: List[SourceDocument] = []


class IngestRequest(BaseModel):
    directory: Optional[str] = Field(
        default=None,
        description="Absolute path to documents directory. Uses DOCS_DIR env var if omitted.",
    )


class IngestResponse(BaseModel):
    status: str
    files: int = 0
    chunks: int = 0
    elapsed_s: float = 0.0


class ResetResponse(BaseModel):
    message: str


class HealthResponse(BaseModel):
    status: str = "ok"
    vector_store: str
    llm_provider: str
    embed_model: str
