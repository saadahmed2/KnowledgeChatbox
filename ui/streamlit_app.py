"""
Streamlit UI — full-featured chat interface with:
  • Sidebar document ingestion (directory path or file upload)
  • Streaming chat window with source citations
  • Chat history management
  • Live status indicators
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow running from repo root: python -m streamlit run ui/streamlit_app.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app import config
from app.chains.rag_chain import ChatSession
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.vector_store import VectorStoreManager

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Knowledge Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stChatMessage { border-radius: 12px; padding: 8px; }
    .source-box {
        background: #f0f2f6; border-left: 4px solid #4c7ef3;
        padding: 8px 12px; border-radius: 4px;
        font-size: 0.82rem; margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session state bootstrap ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_vs_manager() -> VectorStoreManager:
    return VectorStoreManager()


@st.cache_resource(show_spinner=False)
def _get_pipeline() -> IngestionPipeline:
    return IngestionPipeline()


def _get_chat_session() -> ChatSession:
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = ChatSession(_get_vs_manager())
    return st.session_state.chat_session


if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role": "user"|"assistant", "content": str, "sources": []}

if "ingested" not in st.session_state:
    st.session_state.ingested = False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Knowledge Chatbot")
    st.caption("Powered by LangChain + RAG")
    st.divider()

    st.subheader("📂 Ingest Documents")

    ingest_tab, upload_tab = st.tabs(["From Directory", "Upload Files"])

    with ingest_tab:
        docs_dir = st.text_input(
            "Documents directory path",
            value=str(config.DOCS_DIR),
            placeholder="/path/to/your/documents",
        )
        if st.button("▶ Ingest Directory", use_container_width=True, type="primary"):
            with st.spinner("Ingesting documents…"):
                try:
                    stats = _get_pipeline().ingest_directory(docs_dir)
                    st.success(
                        f"✔ Done — {stats['files']} files, "
                        f"{stats['chunks']} chunks in {stats['elapsed_s']}s"
                    )
                    st.session_state.ingested = True
                except FileNotFoundError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")

    with upload_tab:
        uploaded_files = st.file_uploader(
            "Upload documents",
            accept_multiple_files=True,
            type=[ext.lstrip(".") for ext in config.SUPPORTED_EXTENSIONS],
        )
        if st.button("▶ Ingest Uploads", use_container_width=True, type="primary"):
            if not uploaded_files:
                st.warning("Select at least one file first.")
            else:
                import tempfile

                pipeline = _get_pipeline()
                progress = st.progress(0.0, text="Starting…")
                total = len(uploaded_files)
                total_chunks = 0
                for i, uf in enumerate(uploaded_files):
                    suffix = Path(uf.name).suffix
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(uf.read())
                        tmp_path = Path(tmp.name)
                    stats = pipeline.ingest_file(tmp_path)
                    tmp_path.unlink(missing_ok=True)
                    total_chunks += stats.get("chunks", 0)
                    progress.progress((i + 1) / total, text=f"Processed {uf.name}")
                progress.empty()
                st.success(f"✔ {total} files ingested — {total_chunks} total chunks")
                st.session_state.ingested = True

    st.divider()
    st.subheader("⚙️ Settings")
    st.caption(f"**LLM:** `{config.LLM_PROVIDER}` / `{config.OPENAI_CHAT_MODEL if config.LLM_PROVIDER == 'openai' else config.OLLAMA_CHAT_MODEL}`")
    st.caption(f"**Vector store:** `{config.VECTOR_STORE_PROVIDER}`")
    st.caption(f"**Chunk size:** `{config.CHUNK_SIZE}` / overlap `{config.CHUNK_OVERLAP}`")
    st.caption(f"**Top-K:** `{config.RETRIEVER_TOP_K}` ({config.RETRIEVER_SEARCH_TYPE})")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            _get_chat_session().reset_history()
            st.rerun()
    with col2:
        if st.button("♻️ Reset KB", use_container_width=True):
            if st.session_state.get("confirm_reset"):
                _get_pipeline().reset_knowledge_base()
                st.session_state.ingested = False
                st.session_state.confirm_reset = False
                st.success("Knowledge base reset.")
                st.rerun()
            else:
                st.session_state.confirm_reset = True
                st.warning("Click again to confirm reset.")


# ── Main chat area ────────────────────────────────────────────────────────────
st.title("💬 Ask your Questions")

if not st.session_state.ingested:
    st.info(
        "👈 Use the sidebar to **ingest documents** first — "
        "provide a directory path or upload files directly."
    )

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources", expanded=False):
                for src in msg["sources"]:
                    chunk_label = (" — chunk #" + str(src["chunk_index"])) if src.get("chunk_index") else ""
                    st.markdown(
                        f'<div class="source-box">'
                        f'<b>{src["filename"]}</b>'
                        f'{chunk_label}'
                        f'<br><small>{src["content"][:300]}…</small>'
                        '</div>',
                        unsafe_allow_html=True,
                    )

# Input
if question := st.chat_input(
    "Ask a question about your documents…",
    disabled=not st.session_state.ingested,
):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.markdown(question)

    # Stream assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response: list[str] = []
        sources_raw: list = []

        session = _get_chat_session()

        # Synchronous token-by-token streaming — no asyncio.run() needed
        for token, src in session.stream_chat(question):
            if src:
                sources_raw = src   # final sentinel carrying source docs
            else:
                full_response.append(token)
                placeholder.markdown("".join(full_response) + "▌")

        placeholder.markdown("".join(full_response))

        sources_fmt = [
            {
                "filename": d.metadata.get("filename", ""),
                "chunk_index": d.metadata.get("chunk_index"),
                "content": d.page_content[:300],
            }
            for d in sources_raw
        ]

        if sources_fmt:
            with st.expander("📄 Sources", expanded=False):
                for src in sources_fmt:
                    chunk_label = (" — chunk #" + str(src["chunk_index"])) if src.get("chunk_index") else ""
                    st.markdown(
                        f'<div class="source-box">'
                        f'<b>{src["filename"]}</b>'
                        f'{chunk_label}'
                        f'<br><small>{src["content"]}…</small>'
                        '</div>',
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": "".join(full_response),
            "sources": sources_fmt,
        }
    )
