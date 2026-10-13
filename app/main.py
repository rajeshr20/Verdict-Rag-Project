import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.api.routes_ask import router as ask_router
from app.api.routes_documents import router as documents_router
from app.api.routes_health import router as health_router
from app.config import LLMConfigError, Settings, get_settings
from app.core.generation.llm import LLMError, get_llm_provider
from app.core.pipeline import RAGPipeline
from app.core.retrieval.bm25_index import BM25Index
from app.core.retrieval.embedder import get_embedder
from app.core.retrieval.hybrid import HybridRetriever
from app.core.retrieval.vector_store import VectorStore


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("verdict")


def create_app(custom_settings: Optional[Settings] = None) -> FastAPI:
    """Application factory for VERDICT FastAPI service."""
    settings = custom_settings or get_settings()

    # Set log level from settings
    numeric_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: initialize stores, embedder, LLM, and pipeline
        logger.info("Initializing VERDICT RAG service...")
        logger.info("LLM Provider: %s", settings.LLM_PROVIDER)
        logger.info("Embedding Model: %s", settings.EMBEDDING_MODEL)

        app.state.settings = settings
        app.state.vector_store = VectorStore(persist_dir=settings.CHROMA_DIR)
        app.state.embedder = get_embedder(settings.EMBEDDING_MODEL)
        app.state.bm25_index = BM25Index()

        # Pre-populate BM25 index from any existing chunks in persistent vector store
        existing_chunks = app.state.vector_store.get_all_chunks()
        app.state.bm25_index.build(existing_chunks)
        logger.info("Loaded %d existing chunks into BM25 index.", len(existing_chunks))

        app.state.retriever = HybridRetriever(
            vector_store=app.state.vector_store,
            bm25_index=app.state.bm25_index,
            embedder=app.state.embedder,
            rrf_k=60,
        )

        groq_key = (
            settings.GROQ_API_KEY.get_secret_value()
            if hasattr(settings.GROQ_API_KEY, "get_secret_value")
            else str(settings.GROQ_API_KEY)
        )
        app.state.llm = get_llm_provider(
            provider_name=settings.LLM_PROVIDER,
            api_key=groq_key,
            model_name=settings.GROQ_MODEL,
        )

        app.state.pipeline = RAGPipeline(
            retriever=app.state.retriever,
            llm=app.state.llm,
            default_top_k=settings.TOP_K,
            min_score=settings.RETRIEVAL_MIN_SCORE,
        )

        logger.info("VERDICT service startup complete.")
        yield
        logger.info("VERDICT service shutting down.")

    app = FastAPI(
        title="VERDICT API",
        version="1.0.0",
        description="Claim-verified, conflict-aware RAG API with deterministic quote auditing.",
        lifespan=lifespan,
    )

    # Middleware: Request ID and latency logging
    @app.middleware("http")
    async def request_id_and_logging_middleware(request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = req_id
        start_time = time.perf_counter()

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"

        logger.info(
            "[%s] %s %s -> status %d (%.2f ms)",
            req_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    # Exception Handlers: Never leak stack traces
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        req_id = getattr(request.state, "request_id", "unknown")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code,
                "request_id": req_id,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        req_id = getattr(request.state, "request_id", "unknown")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Request validation failed",
                "details": exc.errors(),
                "request_id": req_id,
            },
        )

    @app.exception_handler(LLMConfigError)
    async def llm_config_error_exception_handler(request: Request, exc: LLMConfigError):
        req_id = getattr(request.state, "request_id", "unknown")
        logger.error("[%s] LLM configuration error: %s", req_id, exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "LLM configuration invalid or missing",
                "detail": str(exc),
                "request_id": req_id,
            },
        )

    @app.exception_handler(LLMError)
    async def llm_error_exception_handler(request: Request, exc: LLMError):
        req_id = getattr(request.state, "request_id", "unknown")
        logger.error("[%s] LLM service unavailable: %s", req_id, exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "LLM generation service unavailable",
                "detail": str(exc),
                "request_id": req_id,
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        req_id = getattr(request.state, "request_id", "unknown")
        logger.error("[%s] Unhandled server error: %s", req_id, exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "detail": "An unexpected error occurred while processing the request.",
                "request_id": req_id,
            },
        )

    # Register API Routers
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(ask_router)

    return app


# Module-level default application instance for uvicorn
app = create_app()
