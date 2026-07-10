"""Ingestion pipeline: normalize → embed (batched) → upsert → update CodeIndex,
plus catalog-sync operations (archive / delete / reconcile / diff / stats)."""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import Any

import orjson
from fastapi import HTTPException
from qdrant_client import models

from app.core.config import Settings
from app.models.product import (
    BatchItemResult,
    BatchUpsertResult,
    CatalogStats,
    DiffResult,
    PriceUpdate,
    ProductIn,
    ProductStatus,
    ReconcileResult,
)
from app.services.code_index import CodeIndex
from app.services.embedding import EmbeddingService
from app.services.enrichment import Enrichment, ProductEnrichmentService
from app.services.normalization import compose_dense_text, compose_sparse_text, norm_code
from app.services.qdrant import QdrantService, archived_filter
from app.utils.ids import point_id_for

logger = logging.getLogger(__name__)

# fields that feed the dense/sparse embedding text — changing any of them invalidates
# the stored vectors; price/stock/currency/status do not appear here (payload-only)
_CONTENT_FIELDS = (
    "name",
    "description",
    "brand",
    "category",
    "article",
    "product_code",
    "ean13",
    "attributes",
)


def content_hash(product: ProductIn) -> str:
    """Stable hash of the embedding-affecting fields, used to skip re-embedding a
    product whose searchable content did not change."""
    canonical = orjson.dumps({f: getattr(product, f) for f in _CONTENT_FIELDS}, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(canonical).hexdigest()


def build_payload(
    product: ProductIn, dense_model: str, enrichment: Enrichment | None, hash_: str
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
        "status": ProductStatus.ACTIVE.value,  # present in a batch ⇒ active
        "content_hash": hash_,
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "embed_model": dense_model,
    }
    if enrichment is not None:
        payload["enrichment"] = {
            "normalized_title": enrichment.normalized_title,
            "synonyms": list(enrichment.synonyms),
            "use_cases": list(enrichment.use_cases),
            "spec_summary": enrichment.spec_summary,
        }
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
        """Add/update products. Re-embeds only those whose searchable content changed
        (content-hash short-circuit) — an unchanged re-push just refreshes the payload
        and re-activates the product.

        On failure the whole request raises (no lying partial-success result). Storage is
        not transactional across upsert chunks, so an earlier chunk may already be
        committed to Qdrant when a later one fails; the CodeIndex is updated per committed
        chunk (not once at the end) so it can never drift from Qdrant on a mid-batch
        failure. A retry is safe: committed products skip re-embedding via content_hash."""
        # last write wins for duplicated external_ids inside one batch
        unique: dict[str, ProductIn] = {p.external_id: p for p in items}
        products = list(unique.values())
        point_ids = [point_id_for(p.external_id) for p in products]
        hashes = [content_hash(p) for p in products]

        stored = await self.qdrant.retrieve_payloads(point_ids)
        now = datetime.now(UTC).isoformat(timespec="seconds")

        # split into "content changed / new" (needs embedding) vs "unchanged" (payload only)
        to_embed: list[tuple[ProductIn, str, str]] = []  # (product, point_id, hash)
        to_refresh: list[tuple[ProductIn, str]] = []  # (product, point_id)
        for product, pid, hash_ in zip(products, point_ids, hashes, strict=True):
            prev = stored.get(pid)
            if not self.settings.reembed_unchanged and prev is not None and prev.get("content_hash") == hash_:
                to_refresh.append((product, pid))
            else:
                to_embed.append((product, pid, hash_))

        await self._embed_and_upsert(to_embed)
        for product, pid in to_refresh:
            await self.qdrant.set_payload(
                pid,
                {
                    "price": product.price,
                    "currency": product.currency,
                    "in_stock": product.in_stock,
                    "status": ProductStatus.ACTIVE.value,
                    "updated_at": now,
                },
            )
            # already persisted; keep the index consistent and re-activate a restored
            # product (idempotent). Embedded products are indexed per-chunk inside
            # _embed_and_upsert, right after each chunk commits.
            self._index_codes(product, pid)

        if to_refresh:
            logger.info("upsert: %d embedded, %d unchanged (payload-only)", len(to_embed), len(to_refresh))

        results = [BatchItemResult(external_id=p.external_id, ok=True) for p in items]
        return BatchUpsertResult(total=len(results), succeeded=len(results), failed=0, items=results)

    async def _embed_and_upsert(self, batch: list[tuple[ProductIn, str, str]]) -> None:
        if not batch:
            return
        products = [b[0] for b in batch]
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
                payload=build_payload(product, self.settings.dense_model, enrichment, hash_),
            )
            for (product, pid, hash_), enrichment, dv, sv in zip(
                batch, enrichments, dense_vecs, sparse_vecs, strict=True
            )
        ]
        size = self.settings.upsert_batch_size
        for start in range(0, len(points), size):
            await self.qdrant.upsert_points(points[start : start + size])
            # index only after the chunk is committed, so a later chunk failing can't
            # leave committed points missing from the in-memory CodeIndex until restart
            for product, pid, _hash in batch[start : start + size]:
                self._index_codes(product, pid)

    def _index_codes(self, product: ProductIn, pid: str) -> None:
        self.code_index.add_product(
            pid,
            {"article": product.article, "product_code": product.product_code, "ean13": product.ean13},
        )

    async def update_prices(self, updates: list[PriceUpdate]) -> BatchUpsertResult:
        """Payload-only update of price / availability. No embedding and no CodeIndex
        change — vectors and codes are unaffected by price/stock, so this skips the whole
        heavy ingest pipeline. Partial success: unknown external_ids are reported failed,
        not fatal (a price feed routinely references products not in this catalog)."""
        unique: dict[str, PriceUpdate] = {u.external_id: u for u in updates}
        existing = await self.qdrant.retrieve_existing([point_id_for(eid) for eid in unique])
        now = datetime.now(UTC).isoformat(timespec="seconds")

        results: list[BatchItemResult] = []
        for eid, update in unique.items():
            if point_id_for(eid) not in existing:
                results.append(BatchItemResult(external_id=eid, ok=False, error="Product not found"))
                continue
            await self.qdrant.set_payload(point_id_for(eid), {**update.changed_fields(), "updated_at": now})
            results.append(BatchItemResult(external_id=eid, ok=True))
        return _batch_result(results)

    async def set_archived(self, external_ids: list[str], archived: bool) -> BatchUpsertResult:
        """Archive (hide, retain) or restore products — payload-only, reversible, vectors
        and codes untouched. Unknown external_ids are reported failed, not fatal."""
        status = ProductStatus.ARCHIVED if archived else ProductStatus.ACTIVE
        unique = list(dict.fromkeys(external_ids))
        existing = await self.qdrant.retrieve_existing([point_id_for(eid) for eid in unique])
        now = datetime.now(UTC).isoformat(timespec="seconds")

        results: list[BatchItemResult] = []
        for eid in unique:
            if point_id_for(eid) not in existing:
                results.append(BatchItemResult(external_id=eid, ok=False, error="Product not found"))
                continue
            await self.qdrant.set_payload(point_id_for(eid), {"status": status.value, "updated_at": now})
            results.append(BatchItemResult(external_id=eid, ok=True))
        return _batch_result(results)

    async def delete_products(self, external_ids: list[str]) -> BatchUpsertResult:
        """Bulk hard delete — the only destructive operation. Removes vectors and code
        index entries. Unknown external_ids are reported failed, not fatal."""
        unique = list(dict.fromkeys(external_ids))
        existing = await self.qdrant.retrieve_existing([point_id_for(eid) for eid in unique])
        to_delete = [point_id_for(eid) for eid in unique if point_id_for(eid) in existing]
        await self.qdrant.delete_points(to_delete)
        for pid in to_delete:
            self.code_index.remove_product(pid)
        results = [
            BatchItemResult(
                external_id=eid,
                ok=point_id_for(eid) in existing,
                error=None if point_id_for(eid) in existing else "Product not found",
            )
            for eid in unique
        ]
        return _batch_result(results)

    async def reconcile(
        self, external_ids: list[str], dry_run: bool, max_archived: int | None
    ) -> ReconcileResult:
        """Snapshot reconciliation: archive every stored product NOT in `external_ids`
        (orphans left over from lost delete events). Never deletes — archiving is
        reversible, so a truncated snapshot is recoverable. `max_archived` refuses the
        run if it would archive too many, guarding against a broken source feed.

        Snapshot isolation: only products last modified BEFORE this call started are
        eligible, so a product ingested concurrently (updated_at ≥ reconcile start) is
        never wrongly archived. Build the source snapshot immediately before calling to
        keep the still-unprotected window (snapshot build → this call) small."""
        reconcile_start = datetime.now(UTC).isoformat(timespec="seconds")
        snapshot = set(external_ids)
        stored = await self.qdrant.all_external_ids_status()
        orphans = [
            eid
            for eid, status, updated_at in stored
            if eid not in snapshot
            and status != ProductStatus.ARCHIVED.value
            and (updated_at is None or updated_at < reconcile_start)
        ]
        if max_archived is not None and len(orphans) > max_archived:
            raise HTTPException(
                status_code=400,
                detail=f"reconcile would archive {len(orphans)} products (> max_archived={max_archived}); "
                "refusing — check the source snapshot is complete.",
            )
        if not dry_run:
            now = datetime.now(UTC).isoformat(timespec="seconds")
            for eid in orphans:
                await self.qdrant.set_payload(
                    point_id_for(eid), {"status": ProductStatus.ARCHIVED.value, "updated_at": now}
                )
        return ReconcileResult(dry_run=dry_run, archived_count=len(orphans), external_ids=orphans)

    async def diff(self, external_ids: list[str]) -> DiffResult:
        """Report drift between the source snapshot and the index without mutating."""
        snapshot = set(external_ids)
        stored = {eid for eid, _, _ in await self.qdrant.all_external_ids_status()}
        missing = sorted(snapshot - stored)
        extra = sorted(stored - snapshot)
        return DiffResult(
            missing_in_index=missing,
            extra_in_index=extra,
            missing_count=len(missing),
            extra_count=len(extra),
        )

    async def stats(self) -> CatalogStats:
        total = await self.qdrant.count()
        archived = await self.qdrant.count(archived_filter())
        return CatalogStats(total=total, active=total - archived, archived=archived)

    async def delete_product(self, external_id: str) -> bool:
        point_id = point_id_for(external_id)
        deleted = await self.qdrant.delete_point(point_id)
        if deleted:
            self.code_index.remove_product(point_id)
        return deleted


def _batch_result(results: list[BatchItemResult]) -> BatchUpsertResult:
    succeeded = sum(1 for r in results if r.ok)
    return BatchUpsertResult(
        total=len(results), succeeded=succeeded, failed=len(results) - succeeded, items=results
    )
