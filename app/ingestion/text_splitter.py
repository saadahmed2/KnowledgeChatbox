"""
Text splitting strategy.
Uses RecursiveCharacterTextSplitter as the default with
semantic-aware separators. Swap to SemanticChunker for
embedding-based chunking when using OpenAI embeddings.
"""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Separators ordered from coarsest to finest grain
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split raw documents into chunks suitable for embedding.
    Preserves and enriches metadata on every chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=_SEPARATORS,
        length_function=len,
        add_start_index=True,  # stores char offset in metadata
    )

    chunks = splitter.split_documents(documents)

    # Tag each chunk with its position inside the source file
    source_counters: dict[str, int] = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        source_counters[src] = source_counters.get(src, 0) + 1
        chunk.metadata["chunk_index"] = source_counters[src]

    logger.info(
        "Split %d raw docs → %d chunks (size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        config.CHUNK_SIZE,
        config.CHUNK_OVERLAP,
    )
    return chunks
