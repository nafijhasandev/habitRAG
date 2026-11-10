"""
Entry point for the RAG system.

Usage:
    python main.py ingest          Ingest the PDF into the search index
    python main.py ingest --force  Re-ingest even if data already exists
    python main.py query "..."     Ask a question from the command line
    python main.py serve           Start the web UI server

This file wires together all the components and delegates to the
appropriate subsystem based on the command-line argument.
"""

import argparse
import asyncio
import logging
import sys


def _setup_logging() -> None:
    """Configure logging to show timestamps, level, and module name."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _cmd_ingest(args: argparse.Namespace) -> None:
    """Run the ingestion pipeline to process the PDF."""
    from ingestion.pipeline import run_ingestion

    run_ingestion(force=args.force)


def _cmd_query(args: argparse.Namespace) -> None:
    """Run a single query from the command line."""
    import config
    from core.bm25_index import BM25Index
    from core.context_builder import format_source_location
    from core.embeddings import EmbeddingEngine
    from core.llm_client import LLMClient
    from core.query_engine import QueryEngine
    from core.query_expander import QueryExpander
    from core.reranker import Reranker
    from core.retriever import HybridRetriever
    from core.vector_store import VectorStore

    question = " ".join(args.question)
    if not question.strip():
        print("Please provide a question.")
        sys.exit(1)

    print(f"\nQuestion: {question}\n")
    print("Loading components...")

    embedding_engine = EmbeddingEngine()
    vector_store = VectorStore()
    vector_store.require_cosine()
    bm25_index = BM25Index.load(config.BM25_INDEX_PATH)
    retriever = HybridRetriever(vector_store, bm25_index, embedding_engine)
    reranker = Reranker()
    llm_client = LLMClient()
    query_expander = QueryExpander(llm_client)
    engine = QueryEngine(retriever, reranker, llm_client, query_expander)

    print("Running query pipeline...\n")
    result = asyncio.run(engine.query(question))

    print("=" * 60)
    print(f"Query Type: {result['query_type']}")
    print("=" * 60)
    print(f"\n{result['answer']}\n")
    print("-" * 60)
    print("Sources:")
    for i, source in enumerate(result["sources"], 1):
        meta = source.get("metadata", {})
        score = source.get("rerank_score", 0.0)
        preview = source.get("text_preview", "")
        print(f"\n  [{i}] {format_source_location(meta)} (score: {score:.4f})")
        print(f"      {preview}...")
    print()

    timing = result.get("timing", {})
    if timing:
        print(f"Total time: {timing.get('total', 0):.2f}s "
              f"(retrieve: {timing.get('retrieve', 0):.2f}s, "
              f"rerank: {timing.get('rerank', 0):.2f}s, "
              f"generate: {timing.get('generate', 0):.2f}s)")


def _cmd_serve(args: argparse.Namespace) -> None:
    """Start the web UI server."""
    import uvicorn

    import config

    print(f"\nStarting RAG web server at http://localhost:{config.WEB_PORT}")
    print("Press Ctrl+C to stop.\n")

    uvicorn.run(
        "web.app:app",
        host=config.WEB_HOST,
        port=config.WEB_PORT,
        reload=False,
        log_level="info",
    )


def main() -> None:
    """Parse arguments and dispatch to the appropriate command."""
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="Ask Atomic Habits - RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ingest
    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest the PDF into the search index"
    )
    ingest_parser.add_argument(
        "--force", action="store_true",
        help="Re-ingest even if data already exists"
    )

    # query
    query_parser = subparsers.add_parser(
        "query", help="Ask a question from the command line"
    )
    query_parser.add_argument(
        "question", nargs="+", help="The question to ask"
    )

    # serve
    subparsers.add_parser("serve", help="Start the web UI server")

    args = parser.parse_args()

    if args.command == "ingest":
        _cmd_ingest(args)
    elif args.command == "query":
        _cmd_query(args)
    elif args.command == "serve":
        _cmd_serve(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
