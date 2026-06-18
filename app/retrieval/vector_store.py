"""
Vector store manager — wraps Chroma (default) or FAISS.
Handles creation, persistence, and incremental updates.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from app import config
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _build_embeddings():
    """Factory — returns the correct embedding model based on config."""
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=config.OPENAI_EMBED_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
        )

    elif config.LLM_PROVIDER == "azure_openai":
        from langchain_openai import AzureOpenAIEmbeddings

        return AzureOpenAIEmbeddings(
            azure_deployment=config.AZURE_EMBED_DEPLOYMENT,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_version=config.AZURE_OPENAI_API_VERSION,
            openai_api_key=config.OPENAI_API_KEY,
        )

    elif config.LLM_PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=config.OLLAMA_EMBED_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER}")


class VectorStoreManager:
    """
    Thin façade over the chosen vector store backend.
    Supports Chroma (persistent) and FAISS (in-process, no server needed).
    """

    def __init__(self) -> None:
        self._embeddings = _build_embeddings()
        self._store: VectorStore | None = None
        self._store_path = config.VECTOR_STORE_DIR
        self._store_path.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _get_or_create_chroma(self) -> VectorStore:
        import chromadb
        from langchain_chroma import Chroma

        client = chromadb.PersistentClient(path=str(self._store_path))
        return Chroma(
            client=client,
            collection_name=config.CHROMA_COLLECTION_NAME,
            embedding_function=self._embeddings,
        )

    def _get_or_create_faiss(self) -> VectorStore:
        from langchain_community.vectorstores import FAISS

        index_file = self._store_path / "index.faiss"
        if index_file.exists():
            logger.info("Loading existing FAISS index from %s", self._store_path)
            return FAISS.load_local(
                str(self._store_path),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
        logger.info("No FAISS index found — will create on first add.")
        return None  # type: ignore[return-value]

    def _get_store(self) -> VectorStore:
        if self._store is not None:
            return self._store
        if config.VECTOR_STORE_PROVIDER == "chroma":
            self._store = self._get_or_create_chroma()
        elif config.VECTOR_STORE_PROVIDER == "faiss":
            self._store = self._get_or_create_faiss()
        else:
            raise ValueError(f"Unknown VECTOR_STORE_PROVIDER: {config.VECTOR_STORE_PROVIDER}")
        return self._store

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def add_documents(self, documents: List[Document]) -> None:
        """Embed and persist *documents* into the vector store."""
        if not documents:
            logger.warning("add_documents called with empty list — skipping.")
            return

        if config.VECTOR_STORE_PROVIDER == "faiss":
            from langchain_community.vectorstores import FAISS

            existing = self._get_store()
            if existing is None:
                logger.info("Creating new FAISS index with %d chunks.", len(documents))
                self._store = FAISS.from_documents(documents, self._embeddings)
            else:
                logger.info("Merging %d chunks into existing FAISS index.", len(documents))
                new_store = FAISS.from_documents(documents, self._embeddings)
                self._store.merge_from(new_store)  # type: ignore[union-attr]
            self._store.save_local(str(self._store_path))  # type: ignore[union-attr]
        else:
            store = self._get_store()
            store.add_documents(documents)  # type: ignore[union-attr]

        logger.info("Persisted %d chunks to %s store.", len(documents), config.VECTOR_STORE_PROVIDER)

    def as_retriever(self):
        """Return a configured LangChain retriever."""
        store = self._get_store()
        search_kwargs: dict = {"k": config.RETRIEVER_TOP_K}
        if config.RETRIEVER_SEARCH_TYPE == "mmr":
            search_kwargs["fetch_k"] = config.MMR_FETCH_K

        return store.as_retriever(  # type: ignore[union-attr]
            search_type=config.RETRIEVER_SEARCH_TYPE,
            search_kwargs=search_kwargs,
        )

    def similarity_search(self, query: str, k: int | None = None) -> List[Document]:
        store = self._get_store()
        return store.similarity_search(query, k=k or config.RETRIEVER_TOP_K)  # type: ignore[union-attr]

    def reset(self) -> None:
        """Wipe the persisted vector store and reset in-memory state."""
        self._store = None
        if self._store_path.exists():
            shutil.rmtree(self._store_path)
            self._store_path.mkdir(parents=True, exist_ok=True)
        logger.warning("Vector store wiped: %s", self._store_path)
