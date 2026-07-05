"""Qdrant integration: collection schema, payload indexes, hybrid RRF query."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException

from app.core.config import Settings
from app.models.search import SearchFilters

logger = logging.getLogger(__name__)

META_POINT_ID = 1

# transient transport failures (DNS hiccup, dropped socket) must not surface as 5xx:
# every operation we retry is idempotent (upsert by id, retrieve, query, delete by id)
RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY_S = 0.5


async def _with_retry[T](op: Callable[[], Awaitable[T]], what: str) -> T:
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return await op()
        except ResponseHandlingException:
            if attempt == RETRY_ATTEMPTS - 1:
                raise
            delay = RETRY_BASE_DELAY_S * 2**attempt
            logger.warning("Qdrant %s failed (attempt %d), retrying in %.1fs", what, attempt + 1, delay)
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


PAYLOAD_KEYWORD_INDEXES = ["article_norm", "product_code_norm", "ean13", "brand_norm", "category"]

CODE_PAYLOAD_FIELDS = ["article", "product_code", "ean13"]


class SchemaMismatchError(RuntimeError):
    pass


class QdrantService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.collection = settings.collection_name
        self.meta_collection = settings.meta_collection_name
        # generous timeout: large upsert batches + transient resource pressure must not
        # surface as ConnectTimeout/5xx when Qdrant itself is healthy
        self.client = AsyncQdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=30
        )

    async def close(self) -> None:
        await self.client.close()

    # --- schema ---

    async def ensure_collections(self) -> None:
        if not await self.client.collection_exists(self.collection):
            logger.info("Creating collection %s", self.collection)
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    "dense": models.VectorParams(
                        size=self.settings.dense_dim, distance=models.Distance.COSINE
                    ),
                },
                sparse_vectors_config={
                    # IDF modifier is required for Qdrant/bm25 sparse embeddings
                    "sparse_text": models.SparseVectorParams(modifier=models.Modifier.IDF),
                },
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
            )
        for field in PAYLOAD_KEYWORD_INDEXES:
            await self.client.create_payload_index(
                self.collection, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD
            )
        await self.client.create_payload_index(
            self.collection, field_name="price", field_schema=models.PayloadSchemaType.FLOAT
        )
        await self.client.create_payload_index(
            self.collection, field_name="in_stock", field_schema=models.PayloadSchemaType.BOOL
        )
        await self.client.create_payload_index(
            self.collection,
            field_name="name",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                lowercase=True,
                min_token_len=2,
            ),
        )
        await self._verify_meta()

    async def _verify_meta(self) -> None:
        """Qdrant has no collection-level metadata: a 1-point companion collection records
        which embedding model built the index, so a config change can't silently mix vectors."""
        expected = {
            "dense_model": self.settings.dense_model,
            "dense_dim": self.settings.dense_dim,
            "sparse_model": self.settings.sparse_model,
            "schema_version": self.settings.schema_version,
        }
        if not await self.client.collection_exists(self.meta_collection):
            await self.client.create_collection(
                collection_name=self.meta_collection,
                vectors_config={"stub": models.VectorParams(size=1, distance=models.Distance.DOT)},
            )
            await self.client.upsert(
                self.meta_collection,
                points=[models.PointStruct(id=META_POINT_ID, vector={"stub": [0.0]}, payload=expected)],
            )
            return
        points = await self.client.retrieve(self.meta_collection, ids=[META_POINT_ID])
        if not points:
            await self.client.upsert(
                self.meta_collection,
                points=[models.PointStruct(id=META_POINT_ID, vector={"stub": [0.0]}, payload=expected)],
            )
            return
        stored = {k: (points[0].payload or {}).get(k) for k in expected}
        if stored != expected:
            raise SchemaMismatchError(
                f"Collection '{self.collection}' was built with {stored}, but current config is "
                f"{expected}. Either restore the previous settings or reindex: delete the "
                f"'{self.collection}' and '{self.meta_collection}' collections and re-ingest all products."
            )

    # --- write path ---

    async def upsert_points(self, points: list[models.PointStruct]) -> None:
        await _with_retry(lambda: self.client.upsert(self.collection, points=points, wait=True), "upsert")

    async def delete_point(self, point_id: str) -> bool:
        existing = await _with_retry(
            lambda: self.client.retrieve(self.collection, ids=[point_id], with_payload=False),
            "retrieve",
        )
        if not existing:
            return False
        await _with_retry(
            lambda: self.client.delete(
                self.collection, points_selector=models.PointIdsList(points=[point_id]), wait=True
            ),
            "delete",
        )
        return True

    # --- read path ---

    async def retrieve_payloads(self, point_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not point_ids:
            return {}
        points = await _with_retry(
            lambda: self.client.retrieve(self.collection, ids=point_ids, with_payload=True),
            "retrieve",
        )
        return {str(p.id): p.payload or {} for p in points}

    async def hybrid_query(
        self,
        dense_vector: list[float],
        sparse_vector: models.SparseVector,
        flt: models.Filter | None,
        limit: int,
    ) -> list[models.ScoredPoint]:
        prefetch_limit = max(self.settings.prefetch_limit, limit)
        response = await _with_retry(
            lambda: self.client.query_points(
                collection_name=self.collection,
                prefetch=[
                    models.Prefetch(query=dense_vector, using="dense", limit=prefetch_limit, filter=flt),
                    models.Prefetch(
                        query=sparse_vector, using="sparse_text", limit=prefetch_limit, filter=flt
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            ),
            "query_points",
        )
        return response.points

    async def iter_code_payloads(self) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Payload-only scroll over all points, yielding code fields for CodeIndex bootstrap."""
        offset = None
        while True:
            current_offset = offset
            points, offset = await _with_retry(
                lambda: self.client.scroll(
                    self.collection,
                    limit=1000,
                    offset=current_offset,  # noqa: B023 — invoked immediately by _with_retry
                    with_payload=CODE_PAYLOAD_FIELDS,
                    with_vectors=False,
                ),
                "scroll",
            )
            for p in points:
                yield str(p.id), p.payload or {}
            if offset is None:
                return

    async def count(self) -> int:
        result = await self.client.count(self.collection, exact=True)
        return result.count

    async def ping(self) -> bool:
        """Cheap reachability check for the readiness probe."""
        try:
            return await self.client.collection_exists(self.collection)
        except Exception:
            logger.warning("Qdrant ping failed", exc_info=True)
            return False


def build_filter(filters: SearchFilters | None) -> models.Filter | None:
    if filters is None:
        return None
    must: list[models.Condition] = []
    if filters.brand:
        must.append(
            models.FieldCondition(key="brand_norm", match=models.MatchValue(value=filters.brand.lower()))
        )
    if filters.category:
        must.append(models.FieldCondition(key="category", match=models.MatchValue(value=filters.category)))
    if filters.price_min is not None or filters.price_max is not None:
        must.append(
            models.FieldCondition(
                key="price", range=models.Range(gte=filters.price_min, lte=filters.price_max)
            )
        )
    if filters.in_stock is not None:
        must.append(models.FieldCondition(key="in_stock", match=models.MatchValue(value=filters.in_stock)))
    for key, value in filters.attributes.items():
        if isinstance(value, float) and not isinstance(value, bool):
            # Qdrant match conditions don't accept floats — use a degenerate range
            must.append(
                models.FieldCondition(key=f"attributes.{key}", range=models.Range(gte=value, lte=value))
            )
        else:
            must.append(models.FieldCondition(key=f"attributes.{key}", match=models.MatchValue(value=value)))
    return models.Filter(must=must) if must else None


def payload_matches_filters(payload: dict[str, Any], filters: SearchFilters | None) -> bool:
    """In-app filter check for code-branch hits (they bypass Qdrant's vector query)."""
    if filters is None:
        return True
    if filters.brand and (payload.get("brand_norm") or "") != filters.brand.lower():
        return False
    if filters.category and payload.get("category") != filters.category:
        return False
    price = payload.get("price")
    if filters.price_min is not None and (price is None or price < filters.price_min):
        return False
    if filters.price_max is not None and (price is None or price > filters.price_max):
        return False
    if filters.in_stock is not None and payload.get("in_stock") != filters.in_stock:
        return False
    attrs = payload.get("attributes") or {}
    return all(attrs.get(key) == value for key, value in filters.attributes.items())
