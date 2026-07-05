"""Local dense + sparse embeddings via FastEmbed (ONNX, CPU).

Model profiles isolate model-specific quirks (e5 needs "query:"/"passage:" prefixes,
bge-style models need none) so the rest of the code never knows which model runs.
All fastembed calls are synchronous; async callers go through ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from qdrant_client import models

from app.core.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelProfile:
    doc_prefix: str = ""
    query_prefix: str = ""


MODEL_PROFILES: dict[str, ModelProfile] = {
    "intfloat/multilingual-e5-large": ModelProfile(doc_prefix="passage: ", query_prefix="query: "),
    "BAAI/bge-m3": ModelProfile(),
}


class EmbeddingService:
    """Heavy to construct (downloads/loads ONNX models) — instantiate once in lifespan,
    off the event loop."""

    def __init__(self, settings: Settings):
        # deferred so that importing this module (e.g. in unit tests) never pulls in ONNX
        from fastembed import SparseTextEmbedding, TextEmbedding

        self.settings = settings
        self.profile = MODEL_PROFILES.get(settings.dense_model, ModelProfile())
        logger.info("Loading dense model %s ...", settings.dense_model)
        self._dense = TextEmbedding(model_name=settings.dense_model)
        logger.info(
            "Loading sparse model %s (language=%s) ...", settings.sparse_model, settings.sparse_language
        )
        self._sparse = SparseTextEmbedding(
            model_name=settings.sparse_model, language=settings.sparse_language
        )
        # warmup so the first request doesn't pay ONNX session init costs
        self.embed_dense_query("warmup")
        self.embed_sparse_query("warmup")
        logger.info("Embedding models ready")

    # --- documents ---

    def embed_docs(
        self, dense_texts: list[str], sparse_texts: list[str]
    ) -> tuple[list[list[float]], list[models.SparseVector]]:
        prefixed = [self.profile.doc_prefix + t for t in dense_texts]
        dense = [v.tolist() for v in self._dense.embed(prefixed, batch_size=self.settings.embed_batch_size)]
        sparse = [
            models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
            for e in self._sparse.embed(sparse_texts, batch_size=self.settings.embed_batch_size)
        ]
        return dense, sparse

    async def aembed_docs(
        self, dense_texts: list[str], sparse_texts: list[str]
    ) -> tuple[list[list[float]], list[models.SparseVector]]:
        return await asyncio.to_thread(self.embed_docs, dense_texts, sparse_texts)

    # --- queries ---

    def embed_dense_query(self, text: str) -> list[float]:
        return next(iter(self._dense.query_embed(self.profile.query_prefix + text))).tolist()

    def embed_sparse_query(self, text: str) -> models.SparseVector:
        emb = next(iter(self._sparse.query_embed(text)))
        return models.SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())

    async def aembed_query(
        self, dense_text: str, sparse_text: str
    ) -> tuple[list[float], models.SparseVector]:
        def _both() -> tuple[list[float], models.SparseVector]:
            return self.embed_dense_query(dense_text), self.embed_sparse_query(sparse_text)

        return await asyncio.to_thread(_both)
