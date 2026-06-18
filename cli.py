#!/usr/bin/env python
"""
CLI — run ingestion or start the API server from the command line.

Examples
--------
# Ingest a directory of documents
python -m knowledge_chatbot.cli ingest --dir /path/to/docs

# Start the REST API server
python -m knowledge_chatbot.cli serve

# Start the Streamlit UI
python -m knowledge_chatbot.cli ui
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def cmd_ingest(args: argparse.Namespace) -> None:
    from app.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    stats = pipeline.ingest_directory(args.dir)
    print(f"Ingestion complete: {stats}")


def cmd_serve(_args: argparse.Namespace) -> None:
    import uvicorn
    from app import config

    uvicorn.run(
        "app.api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.API_RELOAD,
        log_level=config.LOG_LEVEL.lower(),
    )


def cmd_ui(_args: argparse.Namespace) -> None:
    ui_path = Path(__file__).parent / "ui" / "streamlit_app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(ui_path)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="knowledge_chatbot", description="Knowledge Chatbot CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest documents from a directory")
    p_ingest.add_argument(
        "--dir", default=None, help="Path to documents directory (uses DOCS_DIR env if omitted)"
    )
    p_ingest.set_defaults(func=cmd_ingest)

    # serve
    p_serve = sub.add_parser("serve", help="Start the FastAPI REST server")
    p_serve.set_defaults(func=cmd_serve)

    # ui
    p_ui = sub.add_parser("ui", help="Launch the Streamlit UI")
    p_ui.set_defaults(func=cmd_ui)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
