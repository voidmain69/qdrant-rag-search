"""Ingestion pipeline: normalize → embed (batched) → upsert → update CodeIndex."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from qdrant_client import models

from app.core.config import Settings
from app.models.product import BatchItemResult, BatchUpsertResult, ProductIn
from app.services.code_index import CodeIndex
from app.services.embedding import EmbeddingService
from app.services.normalization import compose_dense_text, compose_sparse_text, norm_code
from app.services.qdrant import QdrantService
from app.utils.ids import point_id_for

logger = logging.getLogger(__name__)


def build_payload(product: ProductIn, dense_model: str) -> dict[str, Any]:
    return {
        "external_id": product.external_id,
        "name": product.name,
        "description": product.description,
        "brand": product.brand,
        "brand_norm": product.brand.lower() if product.brand else None,
        "category": product.category,
        "article": product.article,
        "article_norm": norm_code(product.article) if product.article else None,
        "product_code": product.product_code,
        "product_code_norm": norm_code(product.product_code) if product.product_code else None,
        "ean13": product.ean13,
        "attributes": product.attributes,
        "price": product.price,
        "currency": product.currency,
        "in_stock": product.in_stock,
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "embed_model": dense_model,
    }


class IngestService:
    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingService,
        qdrant: QdrantService,
        code_index: CodeIndex,
    ):
        self.settings = settings
        self.embedder = embedder
        self.qdrant = qdrant
        self.code_index = code_index

    async def upsert_products(self, items: list[ProductIn]) -> BatchUpsertResult:
        """Embed and upsert a batch. The pipeline is all-or-nothing: any failure raises
        (→ HTTP 5xx) and no partial per-item results are produced."""
        # last write wins for duplicated external_ids inside one batch
        unique: dict[str, ProductIn] = {p.external_id: p for p in items}
        products = list(unique.values())
        point_ids = [point_id_for(p.external_id) for p in products]

        dense_texts = [compose_dense_text(p) for p in products]
        sparse_texts = [compose_sparse_text(p) for p in products]
        dense_vecs, sparse_vecs = await self.embedder.aembed_docs(dense_texts, sparse_texts)

        points = [
            models.PointStruct(
                id=pid,
                vector={"dense": dv, "sparse_text": sv},
                payload=build_payload(p, self.settings.dense_model),
            )
            for p, pid, dv, sv in zip(products, point_ids, dense_vecs, sparse_vecs, strict=True)
        ]

        batch = self.settings.upsert_batch_size
        for start in range(0, len(points), batch):
            await self.qdrant.upsert_points(points[start : start + batch])

        for p, pid in zip(products, point_ids, strict=True):
            self.code_index.add_product(
                pid,
                {"article": p.article, "product_code": p.product_code, "ean13": p.ean13},
            )

        results = [BatchItemResult(external_id=p.external_id, ok=True) for p in items]
        return BatchUpsertResult(total=len(results), succeeded=len(results), failed=0, items=results)

    async def delete_product(self, external_id: str) -> bool:
        point_id = point_id_for(external_id)
        deleted = await self.qdrant.delete_point(point_id)
        if deleted:
            self.code_index.remove_product(point_id)
        return deleted
