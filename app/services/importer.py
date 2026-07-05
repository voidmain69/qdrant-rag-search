"""File import (CSV / JSON / XLSX) → ProductIn stream → IngestService.

Runs as a FastAPI BackgroundTask; progress lives in an in-memory JobStore — an accepted
single-instance tradeoff (documented in README).
"""

from __future__ import annotations

import csv
import io
import logging
import uuid
from typing import Any

import orjson
from pydantic import ValidationError

from app.models.imports import ImportJob, JobStatus, RowError
from app.models.product import ProductIn
from app.services.ingest import IngestService

logger = logging.getLogger(__name__)

MAX_ROW_ERRORS = 100
INGEST_CHUNK = 200

# Common source-column names mapped onto ProductIn fields.
DEFAULT_ALIASES = {
    "id": "external_id",
    "код": "product_code",
    "code": "product_code",
    "sku": "article",
    "артикул": "article",
    "ean": "ean13",
    "barcode": "ean13",
    "штрихкод": "ean13",
    "назва": "name",
    "название": "name",
    "опис": "description",
    "описание": "description",
    "бренд": "brand",
    "категорія": "category",
    "категория": "category",
    "ціна": "price",
    "цена": "price",
}

PRODUCT_FIELDS = set(ProductIn.model_fields.keys())


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, ImportJob] = {}

    def create(self, filename: str) -> ImportJob:
        job = ImportJob(job_id=uuid.uuid4().hex, filename=filename)
        self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> ImportJob | None:
        return self._jobs.get(job_id)


def _map_row(row: dict[str, Any], mapping: dict[str, str] | None) -> dict[str, Any]:
    """Known columns map to ProductIn fields (explicit mapping > aliases > exact name);
    everything else lands in attributes."""
    out: dict[str, Any] = {}
    attributes: dict[str, Any] = {}
    for key, value in row.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        key_str = str(key).strip()
        lower = key_str.lower()
        if mapping and key_str in mapping:
            field = mapping[key_str]
        elif lower in PRODUCT_FIELDS:
            field = lower
        elif lower in DEFAULT_ALIASES:
            field = DEFAULT_ALIASES[lower]
        else:
            attributes[key_str] = value if isinstance(value, int | float | bool) else str(value)
            continue
        out[field] = value
    if attributes:
        merged = out.get("attributes") or {}
        if isinstance(merged, dict):
            attributes.update(merged)
        out["attributes"] = attributes
    return out


def parse_rows(filename: str, content: bytes) -> list[dict[str, Any]]:
    lower = filename.lower()
    if lower.endswith(".json"):
        data = orjson.loads(content)
        if not isinstance(data, list):
            raise ValueError("JSON import must be an array of product objects")
        return data
    if lower.endswith(".csv"):
        text = content.decode("utf-8-sig")
        return list(csv.DictReader(io.StringIO(text)))
    if lower.endswith((".xlsx", ".xlsm")):
        from openpyxl import load_workbook

        wb = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
        ws = wb.active
        rows_iter = ws.iter_rows(values_only=True)
        header = next(rows_iter, None)
        if header is None:
            return []
        keys = [str(h) if h is not None else f"col{i}" for i, h in enumerate(header)]
        return [dict(zip(keys, row, strict=False)) for row in rows_iter]
    raise ValueError(f"Unsupported file type: {filename} (expected .json, .csv or .xlsx)")


async def run_import(
    job: ImportJob,
    content: bytes,
    mapping: dict[str, str] | None,
    ingest: IngestService,
) -> None:
    job.status = JobStatus.RUNNING
    try:
        raw_rows = parse_rows(job.filename, content)
    except Exception as exc:
        job.status = JobStatus.FAILED
        job.detail = f"Cannot parse file: {exc}"
        logger.exception("Import %s failed to parse", job.job_id)
        return

    job.total = len(raw_rows)
    products: list[ProductIn] = []
    for i, row in enumerate(raw_rows, start=1):
        try:
            products.append(ProductIn.model_validate(_map_row(row, mapping)))
        except (ValidationError, TypeError, ValueError) as exc:
            job.failed += 1
            if len(job.errors) < MAX_ROW_ERRORS:
                job.errors.append(RowError(row=i, error=str(exc)))

    try:
        for start in range(0, len(products), INGEST_CHUNK):
            chunk = products[start : start + INGEST_CHUNK]
            await ingest.upsert_products(chunk)
            job.processed += len(chunk)
        job.status = JobStatus.COMPLETED
    except Exception as exc:
        job.status = JobStatus.FAILED
        job.detail = f"Ingestion error after {job.processed} rows: {exc}"
        logger.exception("Import %s failed during ingestion", job.job_id)
