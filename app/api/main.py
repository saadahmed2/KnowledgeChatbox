"""
FastAPI application entry point.
Registers all routers, CORS middleware, global exception handlers,
and the /health liveness probe.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import config
from app.api.routes import chat, ingestion
from app.api.schemas import HealthResponse
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Knowledge Chatbot API",
    description=(
        "Production-ready RAG (Retrieval-Augmented Generation) chatbot. "
        "Ingest documents from any directory and ask questions in natural language."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(chat.router)
app.include_router(ingestion.router)


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s — %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Check server logs."},
    )


# ── Health / liveness probe ───────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    embed_model = (
        config.OPENAI_EMBED_MODEL
        if config.LLM_PROVIDER == "openai"
        else config.AZURE_EMBED_DEPLOYMENT
        if config.LLM_PROVIDER == "azure_openai"
        else config.OLLAMA_EMBED_MODEL
    )
    return HealthResponse(
        status="ok",
        vector_store=config.VECTOR_STORE_PROVIDER,
        llm_provider=config.LLM_PROVIDER,
        embed_model=embed_model,
    )


# ── Dev server entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.API_RELOAD,
        log_level=config.LOG_LEVEL.lower(),
    )
