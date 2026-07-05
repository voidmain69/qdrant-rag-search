import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.health import router as health_router
from app.api.v1.router import api_v1
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.services.code_index import CodeIndex
from app.services.embedding import EmbeddingService
from app.services.importer import JobStore
from app.services.ingest import IngestService
from app.services.qdrant import QdrantService
from app.services.query_router import SearchService
from app.services.reranker import RerankerService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.debug)

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

    app.state.qdrant = qdrant
    app.state.embedder = embedder
    app.state.code_index = code_index
    app.state.ingest_service = IngestService(settings, embedder, qdrant, code_index)
    app.state.search_service = SearchService(settings, qdrant, embedder, code_index, reranker)
    app.state.job_store = JobStore()
    app.state.ready = True
    logger.info("Service ready")

    yield

    await qdrant.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Qdrant Product Search",
        version="0.1.0",
        description="Hybrid product search: semantic + BM25 + exact/fuzzy code matching",
        lifespan=lifespan,
        debug=settings.debug,
    )
    app.include_router(health_router)
    app.include_router(api_v1)
    return app


app = create_app()
