"""
FastAPI web application for the RAG system.

Serves a minimal chat interface and exposes a streaming query API.
The frontend communicates via Server-Sent Events (SSE) for real-time
token-by-token response streaming.
"""

import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, StringConstraints

import config
from core.bm25_index import BM25Index
from core.embeddings import EmbeddingEngine
from core.llm_client import LLMClient
from core.query_engine import QueryEngine
from core.query_expander import QueryExpander
from core.reranker import Reranker
from core.retriever import HybridRetriever
from core.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Global references set during startup
_engine: QueryEngine | None = None
_llm_client: LLMClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize all RAG components on startup, clean up on shutdown.

    Using the lifespan context manager (FastAPI's recommended pattern)
    rather than on_event decorators which are deprecated.
    """
    global _engine, _llm_client

    logger.info("Loading RAG components...")

    embedding_engine = EmbeddingEngine()
    vector_store = VectorStore()
    vector_store.require_cosine()

    if not vector_store.collection_exists():
        logger.error(
            "No ingested data found. Run 'python main.py ingest' first."
        )
        raise RuntimeError("No ingested data. Run 'python main.py ingest' first.")

    bm25_index = BM25Index.load(config.BM25_INDEX_PATH)
    retriever = HybridRetriever(vector_store, bm25_index, embedding_engine)
    reranker = Reranker()
    _llm_client = LLMClient()
    query_expander = QueryExpander(_llm_client)

    _engine = QueryEngine(retriever, reranker, _llm_client, query_expander)
    logger.info("All RAG components loaded successfully.")

    yield

    # Cleanup
    if _llm_client is not None:
        await _llm_client.close()
    logger.info("RAG system shut down.")


app = FastAPI(title="Ask Atomic Habits", lifespan=lifespan)

# Serve static files (CSS, JS)
app.mount(
    "/static",
    StaticFiles(directory=str(config.PROJECT_ROOT / "web" / "static")),
    name="static",
)

templates = Jinja2Templates(directory=str(config.PROJECT_ROOT / "web" / "templates"))


class QueryRequest(BaseModel):
    """Request body for the query endpoint."""
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)
    ]
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=config.RERANK_CANDIDATES,
        strict=True,
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Serve the chat interface."""
    return templates.TemplateResponse(request, "index.html")


@app.get("/book.pdf")
async def book_pdf() -> FileResponse:
    """Serve the configured book so citations can open its PDF pages."""
    return FileResponse(config.PDF_PATH, media_type="application/pdf")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Health check endpoint."""
    llm_configured = False
    if _llm_client is not None:
        llm_configured = await _llm_client.is_available()

    chunk_count = 0
    try:
        vs = VectorStore()
        chunk_count = vs.count()
    except Exception:
        logger.exception("Could not read ChromaDB chunk count")

    return {
        "status": "ok",
        "llm_configured": llm_configured,
        "index_loaded": _engine is not None,
        "chunk_count": chunk_count,
    }


@app.post("/api/query")
async def query_endpoint(req: QueryRequest) -> StreamingResponse:
    """Stream a RAG response as Server-Sent Events.

    SSE format:
        data: {"type": "metadata", "query_type": "..."}
        data: {"type": "chunk", "content": "..."}
        data: {"type": "sources", "sources": [...]}
        data: {"type": "done", "answer": "...", "timing": {...}}
        data: {"type": "error", "message": "..."}
    """
    if _engine is None:
        async def error_stream() -> AsyncGenerator[str, None]:
            yield f"data: {json.dumps({'type': 'error', 'message': 'System not initialized'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            async for event in _engine.query_stream(req.question, top_k=req.top_k):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception:
            logger.exception("Query streaming failed")
            yield f"data: {json.dumps({'type': 'error', 'message': 'Query failed. Please try again.'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
