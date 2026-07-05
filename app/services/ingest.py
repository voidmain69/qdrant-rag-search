"""Ingestion pipeline: normalize → embed (batched) → upsert → update CodeIndex."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from qdrant_client import models

from app.core.config import Settings
from app.models.product import BatchItemResult, BatchUpsertResult, PriceUpdate, ProductIn
from app.services.code_index import CodeIndex
from app.services.embedding import EmbeddingService
from app.services.enrichment import Enrichment, ProductEnrichmentService
from app.services.normalization import compose_dense_text, compose_sparse_text, norm_code
from app.services.qdrant import QdrantService
from app.utils.ids import point_id_for

logger = logging.getLogger(__name__)


def build_payload(
    product: ProductIn, dense_model: str, enrichment: Enrichment | None = None
) -> dict[str, Any]:
    # product-provided attributes always win: supplier data is ground truth,
    # LLM enrichment only fills gaps
    attributes = {**(enrichment.attributes if enrichment else {}), **product.attributes}
    payload: dict[str, Any] = {
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
        "attributes": attributes,
        "price": product.price,
        "currency": product.currency,
        "in_stock": product.in_stock,
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "embed_model": dense_model,
    }
    if enrichment is not None:
        enrichment_block: dict[str, Any] = {
            "normalized_title": enrichment.normalized_title,
            "synonyms": list(enrichment.synonyms),
            "use_cases": list(enrichment.use_cases),
            "spec_summary": enrichment.spec_summary,
        }
        payload["enrichment"] = enrichment_block
    return payload


class IngestService:
    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingService,
        qdrant: QdrantService,
        code_index: CodeIndex,
        enricher: ProductEnrichmentService | None = None,
    ):
        self.settings = settings
        self.embedder = embedder
        self.qdrant = qdrant
        self.code_index = code_index
        self.enricher = enricher

    async def upsert_products(self, items: list[ProductIn]) -> BatchUpsertResult:
        """Embed and upsert a batch. The pipeline is all-or-nothing: any failure raises
        (→ HTTP 5xx) and no partial per-item results are produced."""
        # last write wins for duplicated external_ids inside one batch
        unique: dict[str, ProductIn] = {p.external_id: p for p in items}
        products = list(unique.values())
        point_ids = [point_id_for(p.external_id) for p in products]

        enrichments: list[Enrichment | None]
        if self.enricher is not None:
            enrichments = await self.enricher.enrich_all(products)
        else:
            enrichments = [None] * len(products)

        dense_texts = [compose_dense_text(p, e) for p, e in zip(products, enrichments, strict=True)]
        sparse_texts = [compose_sparse_text(p, e) for p, e in zip(products, enrichments, strict=True)]
        dense_vecs, sparse_vecs = await self.embedder.aembed_docs(dense_texts, sparse_texts)

        points = [
            models.PointStruct(
                id=pid,
                vector={"dense": dv, "sparse_text": sv},
                payload=build_payload(p, self.settings.dense_model, e),
            )
            for p, pid, dv, sv, e in zip(
                products, point_ids, dense_vecs, sparse_vecs, enrichments, strict=True
            )
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

    async def update_prices(self, updates: list[PriceUpdate]) -> BatchUpsertResult:
        """Payload-only update of price / availability. No embedding and no CodeIndex
        change — vectors and codes are unaffected by price/stock, so this skips the whole
        heavy ingest pipeline. Partial success: unknown external_ids are reported failed,
        not fatal (a price feed routinely references products not in this catalog)."""
        # last write wins for duplicated external_ids inside one batch
        unique: dict[str, PriceUpdate] = {u.external_id: u for u in updates}
        point_ids = {point_id_for(eid): eid for eid in unique}
        existing = await self.qdrant.retrieve_existing(list(point_ids))
        now = datetime.now(UTC).isoformat(timespec="seconds")

        results: list[BatchItemResult] = []
        for eid, update in unique.items():
            if point_id_for(eid) not in existing:
                results.append(BatchItemResult(external_id=eid, ok=False, error="Product not found"))
                continue
            await self.qdrant.set_payload(point_id_for(eid), {**update.changed_fields(), "updated_at": now})
            results.append(BatchItemResult(external_id=eid, ok=True))

        succeeded = sum(1 for r in results if r.ok)
        return BatchUpsertResult(
            total=len(results), succeeded=succeeded, failed=len(results) - succeeded, items=results
        )

    async def delete_product(self, external_id: str) -> bool:
        point_id = point_id_for(external_id)
        deleted = await self.qdrant.delete_point(point_id)
        if deleted:
            self.code_index.remove_product(point_id)
        return deleted
