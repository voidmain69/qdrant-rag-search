from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.api.deps import get_ingest_service, get_job_store
from app.models.imports import ImportJob
from app.models.product import (
    ArchiveRequest,
    BatchUpsertRequest,
    BatchUpsertResult,
    CatalogStats,
    DeleteRequest,
    DiffRequest,
    DiffResult,
    PriceUpdate,
    PriceUpdateBatch,
    ProductIn,
    ReconcileRequest,
    ReconcileResult,
)
from app.services.importer import JobStore, run_batch
from app.services.ingest import IngestService

router = APIRouter(tags=["products"])


@router.post("/products", response_model=BatchUpsertResult)
async def upsert_product(
    product: ProductIn, ingest: IngestService = Depends(get_ingest_service)
) -> BatchUpsertResult:
    return await ingest.upsert_products([product])


@router.post("/products:batch", response_model=BatchUpsertResult)
async def upsert_products_batch(
    body: BatchUpsertRequest, ingest: IngestService = Depends(get_ingest_service)
) -> BatchUpsertResult:
    """Synchronous batch upsert — returns once every item is embedded and stored. Fine for
    modest batches; when ingest enrichment is on a large batch can run for minutes and time
    out the request — use `POST /products:batch-async` for those."""
    return await ingest.upsert_products(body.items)


@router.post("/products:batch-async", response_model=ImportJob, status_code=202)
async def upsert_products_batch_async(
    body: BatchUpsertRequest,
    background: BackgroundTasks,
    ingest: IngestService = Depends(get_ingest_service),
    jobs: JobStore = Depends(get_job_store),
) -> ImportJob:
    """Enqueue a batch upsert as a background job (202 + job_id); poll
    `GET /imports/{job_id}` for progress. The same heavy pipeline as `:batch`, but it can't
    time out the request — for large or enrichment-heavy batches. Job state is durable when
    JOBS_DB_PATH is set, so progress survives a restart."""
    job = await jobs.create(f"batch upsert ({len(body.items)} items)")
    background.add_task(run_batch, job, body.items, ingest, jobs)
    return job


@router.put("/products/{external_id}", response_model=BatchUpsertResult)
async def replace_product(
    external_id: str, product: ProductIn, ingest: IngestService = Depends(get_ingest_service)
) -> BatchUpsertResult:
    if product.external_id != external_id:
        raise HTTPException(status_code=422, detail="external_id in path and body must match")
    return await ingest.upsert_products([product])


@router.patch("/products/{external_id}/price", response_model=BatchUpsertResult)
async def update_price(
    external_id: str, update: PriceUpdate, ingest: IngestService = Depends(get_ingest_service)
) -> BatchUpsertResult:
    """Update only price / availability — no re-embedding (vectors are unaffected)."""
    if update.external_id != external_id:
        raise HTTPException(status_code=422, detail="external_id in path and body must match")
    result = await ingest.update_prices([update])
    if result.failed:
        raise HTTPException(status_code=404, detail=f"Product '{external_id}' not found")
    return result


@router.post("/products:prices", response_model=BatchUpsertResult)
async def update_prices_batch(
    body: PriceUpdateBatch, ingest: IngestService = Depends(get_ingest_service)
) -> BatchUpsertResult:
    """Bulk price / availability update (up to 1000). Partial success: unknown
    external_ids come back as failed items rather than failing the whole request."""
    return await ingest.update_prices(body.items)


@router.delete("/products/{external_id}", status_code=204)
async def delete_product(external_id: str, ingest: IngestService = Depends(get_ingest_service)) -> None:
    if not await ingest.delete_product(external_id):
        raise HTTPException(status_code=404, detail=f"Product '{external_id}' not found")


# --- catalog sync / lifecycle ---


@router.post("/products:archive", response_model=BatchUpsertResult)
async def archive_products(
    body: ArchiveRequest, ingest: IngestService = Depends(get_ingest_service)
) -> BatchUpsertResult:
    """Archive (hide, retain) or restore products (`archived: false`). Reversible,
    payload-only — vectors are kept. Partial success on unknown external_ids."""
    return await ingest.set_archived(body.external_ids, body.archived)


@router.post("/products:delete", response_model=BatchUpsertResult)
async def delete_products_bulk(
    body: DeleteRequest, ingest: IngestService = Depends(get_ingest_service)
) -> BatchUpsertResult:
    """Bulk hard delete — removes vectors permanently. Partial success on unknown ids."""
    return await ingest.delete_products(body.external_ids)


@router.post("/products:reconcile", response_model=ReconcileResult)
async def reconcile_catalog(
    body: ReconcileRequest, ingest: IngestService = Depends(get_ingest_service)
) -> ReconcileResult:
    """Snapshot reconciliation: archive every product NOT in `external_ids` (never
    deletes). `dry_run` (default true) reports what would change; `max_archived` refuses
    the run past a threshold (400) to guard against a truncated source snapshot."""
    return await ingest.reconcile(body.external_ids, body.dry_run, body.max_archived)


@router.post("/products:diff", response_model=DiffResult)
async def diff_catalog(body: DiffRequest, ingest: IngestService = Depends(get_ingest_service)) -> DiffResult:
    """Drift report between the source snapshot and the index (no mutation)."""
    return await ingest.diff(body.external_ids)


@router.get("/products/stats", response_model=CatalogStats)
async def catalog_stats(ingest: IngestService = Depends(get_ingest_service)) -> CatalogStats:
    """Counts: total / active / archived — a cheap divergence check for the source."""
    return await ingest.stats()
