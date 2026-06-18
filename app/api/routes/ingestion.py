"""
/ingest  endpoints — trigger ingestion from directory or single file upload.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.api.schemas import IngestRequest, IngestResponse, ResetResponse
from app.ingestion.pipeline import IngestionPipeline
from app import config
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/ingest", tags=["Ingestion"])

_pipeline = IngestionPipeline()


@router.post("/directory", response_model=IngestResponse, summary="Ingest a document directory")
async def ingest_directory(request: IngestRequest):
    """
    Recursively load, chunk, embed and store all documents
    found in *directory* (or $DOCS_DIR if omitted).
    """
    directory = request.directory or str(config.DOCS_DIR)
    try:
        stats = _pipeline.ingest_directory(directory)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Ingestion error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    return IngestResponse(**stats)


@router.post("/file", response_model=IngestResponse, summary="Upload and ingest a single file")
async def ingest_file(file: UploadFile = File(...)):
    """
    Accept a file upload, persist it to a temp location,
    ingest it into the knowledge base, then clean up.
    """
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in config.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Supported: {config.SUPPORTED_EXTENSIONS}",
        )
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        stats = _pipeline.ingest_file(tmp_path)
        stats["file"] = file.filename  # override with original name
    except Exception as exc:
        logger.error("File ingestion error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return IngestResponse(**{k: v for k, v in stats.items() if k in IngestResponse.model_fields})


@router.delete("/reset", response_model=ResetResponse, summary="Reset the knowledge base")
async def reset_knowledge_base():
    """
    ⚠️  Wipes all vectors and starts fresh.
    You must re-ingest all documents after calling this endpoint.
    """
    try:
        _pipeline.reset_knowledge_base()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return ResetResponse(message="Knowledge base reset. Please re-ingest your documents.")
