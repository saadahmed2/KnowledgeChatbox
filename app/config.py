"""
Central configuration — reads from environment variables / .env file.
All tuneable parameters live here so nothing is hard-coded elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Resolve .env relative to this file: knowledge_chatbot/.env
_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=_ENV_FILE, override=True)


# ──────────────────────────────────────────────
# LLM / Embedding provider
# ──────────────────────────────────────────────
LLM_PROVIDER: Literal["openai", "azure_openai", "ollama"] = os.getenv(
    "LLM_PROVIDER", "openai"
)

# OpenAI / Azure OpenAI
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
OPENAI_EMBED_MODEL: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Azure-specific overrides (only used when LLM_PROVIDER == "azure_openai")
AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_CHAT_DEPLOYMENT: str = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o")
AZURE_EMBED_DEPLOYMENT: str = os.getenv("AZURE_EMBED_DEPLOYMENT", "text-embedding-3-small")

# Ollama (local, no API key required)
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3")
OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# ──────────────────────────────────────────────
# Vector store
# ──────────────────────────────────────────────
VECTOR_STORE_PROVIDER: Literal["chroma", "faiss"] = os.getenv(
    "VECTOR_STORE_PROVIDER", "chroma"
)
VECTOR_STORE_DIR: Path = Path(
    os.getenv("VECTOR_STORE_DIR", "./knowledge_chatbot/vector_store")
)
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "knowledge_base")

# ──────────────────────────────────────────────
# Document ingestion
# ──────────────────────────────────────────────
DOCS_DIR: Path = Path(os.getenv("DOCS_DIR", "./knowledge_chatbot/documents"))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
SUPPORTED_EXTENSIONS: list[str] = [
    ".pdf", ".docx", ".doc", ".txt", ".md",
    ".csv", ".xlsx", ".xls", ".html", ".htm", ".pptx",
]

# ──────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────
RETRIEVER_TOP_K: int = int(os.getenv("RETRIEVER_TOP_K", "5"))
# "similarity" | "mmr" | "similarity_score_threshold"
RETRIEVER_SEARCH_TYPE: str = os.getenv("RETRIEVER_SEARCH_TYPE", "mmr")
MMR_FETCH_K: int = int(os.getenv("MMR_FETCH_K", "20"))

# ──────────────────────────────────────────────
# RAG chain / prompting
# ──────────────────────────────────────────────
MAX_CHAT_HISTORY: int = int(os.getenv("MAX_CHAT_HISTORY", "10"))  # message pairs
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# ──────────────────────────────────────────────
# API server
# ──────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
API_RELOAD: bool = os.getenv("API_RELOAD", "false").lower() == "true"
CORS_ORIGINS: list[str] = os.getenv("CORS_ORIGINS", "*").split(",")

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: str = os.getenv("LOG_FILE", "./knowledge_chatbot/logs/app.log")
