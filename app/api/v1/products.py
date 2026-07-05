from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_ingest_service
from app.models.product import BatchUpsertRequest, BatchUpsertResult, ProductIn
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
    return await ingest.upsert_products(body.items)


@router.put("/products/{external_id}", response_model=BatchUpsertResult)
async def replace_product(
    external_id: str, product: ProductIn, ingest: IngestService = Depends(get_ingest_service)
) -> BatchUpsertResult:
    if product.external_id != external_id:
        raise HTTPException(status_code=422, detail="external_id in path and body must match")
    return await ingest.upsert_products([product])


@router.delete("/products/{external_id}", status_code=204)
async def delete_product(external_id: str, ingest: IngestService = Depends(get_ingest_service)) -> None:
    if not await ingest.delete_product(external_id):
        raise HTTPException(status_code=404, detail=f"Product '{external_id}' not found")
