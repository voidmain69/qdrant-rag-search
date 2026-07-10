import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.v1.health import router as health_router
from app.api.v1.router import api_v1
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.monitoring import init_sentry, setup_metrics
from app.services.code_index import CodeIndex
from app.services.embedding import EmbeddingService
from app.services.enrichment import ProductEnrichmentService
from app.services.importer import JobStore
from app.services.ingest import IngestService
from app.services.llm import build_llm_client
from app.services.qdrant import QdrantService
from app.services.query_router import SearchService
from app.services.query_understanding import QueryUnderstandingService
from app.services.reranker import RerankerService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    # heavy: downloads/loads ONNX models — keep it off the event loop
    embedder = await asyncio.to_thread(EmbeddingService, settings)

    qdrant = QdrantService(settings)
    await qdrant.ensure_collections()

    code_index = CodeIndex()
    async for point_id, payload in qdrant.iter_code_payloads():
        code_index.add_product(
            point_id,
            {
                "article": payload.get("article"),
                "product_code": payload.get("product_code"),
                "ean13": payload.get("ean13"),
            },
        )
    logger.info("CodeIndex bootstrapped with %d products", len(code_index))

    reranker = RerankerService(settings.rerank_model) if settings.rerank_enabled else None

    # one LLM backend (Ollama or OpenAI-compatible) shared by both LLM features
    llm_client = build_llm_client(settings) if settings.llm_enabled else None
    provider = settings.llm_provider.value

    understanding = (
        QueryUnderstandingService(llm_client, settings) if llm_client and settings.query_llm_enabled else None
    )
    if understanding:
        logger.info("LLM query understanding enabled (%s via %s)", settings.query_llm_model, provider)
        warmup_task = asyncio.create_task(understanding.warmup())
        warmup_task.add_done_callback(lambda _t: None)  # keep a reference until it completes

    enricher = (
        ProductEnrichmentService(llm_client, settings) if llm_client and settings.ingest_llm_enabled else None
    )
    if enricher:
        logger.info("LLM ingest enrichment enabled (%s via %s)", settings.ingest_llm_model, provider)

    app.state.qdrant = qdrant
    app.state.embedder = embedder
    app.state.code_index = code_index
    app.state.ingest_service = IngestService(settings, embedder, qdrant, code_index, enricher)
    app.state.search_service = SearchService(settings, qdrant, embedder, code_index, reranker, understanding)
    job_store = JobStore(settings.jobs_db_path)
    await job_store.initialize()  # load persisted jobs + reap ones a dead process left running
    app.state.job_store = job_store
    app.state.ready = True
    logger.info("Service ready")

    yield

    if llm_client:
        await llm_client.aclose()
    job_store.close()
    await qdrant.close()


async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Sentry (if enabled) captures the exception via its ASGI integration; here we make
    # sure it is also logged with the request id and the client gets structured JSON.
    logger.error("Unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings)
    init_sentry(settings)

    app = FastAPI(
        title="Qdrant Product Search",
        version="0.1.0",
        description="Hybrid product search: semantic + BM25 + exact/fuzzy code matching",
        lifespan=lifespan,
        debug=settings.debug,
    )
    app.add_middleware(CorrelationIdMiddleware)
    app.add_exception_handler(Exception, _unhandled_exception_handler)
    if settings.metrics_enabled:
        setup_metrics(app)
    app.include_router(health_router)
    app.include_router(api_v1)
    return app


app = create_app()
