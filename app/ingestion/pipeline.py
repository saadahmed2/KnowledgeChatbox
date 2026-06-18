"""
Ingestion pipeline — orchestrates load → split → embed → store.
Can be triggered via CLI, the REST API, or the Streamlit UI.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from app import config
from app.ingestion.document_loader import load_directory, load_document
from app.ingestion.text_splitter import split_documents
from app.retrieval.vector_store import VectorStoreManager
from app.utils.logger import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    """
    Encapsulates the full ETL flow for document ingestion.

    Usage
    -----
    pipeline = IngestionPipeline()
    stats = pipeline.ingest_directory("/path/to/docs")
    """

    def __init__(self) -> None:
        self.vs_manager = VectorStoreManager()

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def ingest_directory(self, directory: Optional[str | Path] = None) -> dict:
        """
        Load all supported documents from *directory*, chunk, embed and store them.

        Returns a stats dict with counts and timing.
        """
        directory = Path(directory or config.DOCS_DIR)
        start = time.perf_counter()

        logger.info("▶ Ingestion started — source: %s", directory)

        raw_docs = load_directory(directory)
        if not raw_docs:
            logger.warning("No documents found in: %s", directory)
            return {"status": "empty", "files": 0, "chunks": 0, "elapsed_s": 0}

        chunks = split_documents(raw_docs)

        unique_sources = {d.metadata.get("source") for d in raw_docs}
        self.vs_manager.add_documents(chunks)

        elapsed = round(time.perf_counter() - start, 2)
        stats = {
            "status": "success",
            "files": len(unique_sources),
            "chunks": len(chunks),
            "elapsed_s": elapsed,
        }
        logger.info("✔ Ingestion complete — %s", stats)
        return stats

    def ingest_file(self, file_path: str | Path) -> dict:
        """
        Ingest a single file on demand (e.g., uploaded via the API).
        """
        file_path = Path(file_path)
        start = time.perf_counter()

        raw_docs = load_document(file_path)
        if not raw_docs:
            return {"status": "unsupported_or_empty", "chunks": 0}

        chunks = split_documents(raw_docs)
        self.vs_manager.add_documents(chunks)

        elapsed = round(time.perf_counter() - start, 2)
        stats = {
            "status": "success",
            "file": file_path.name,
            "chunks": len(chunks),
            "elapsed_s": elapsed,
        }
        logger.info("✔ File ingestion complete — %s", stats)
        return stats

    def reset_knowledge_base(self) -> None:
        """Delete and recreate the vector store (full re-index)."""
        logger.warning("Resetting knowledge base — all vectors will be deleted.")
        self.vs_manager.reset()
        logger.info("Knowledge base reset complete.")
