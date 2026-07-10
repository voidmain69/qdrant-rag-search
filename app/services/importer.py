"""File import (CSV / JSON / XLSX) and async batch upsert → ProductIn stream → IngestService.

Both run as FastAPI BackgroundTasks; progress lives in a :class:`JobStore`. The store is
durable when `JOBS_DB_PATH` is set (SQLite): jobs survive a restart, and a job left
RUNNING by a process that died is reaped to FAILED on the next boot. With no path it stays
purely in-memory (the pre-durability behavior) — jobs then vanish on restart.
"""

from __future__ import annotations

import asyncio
import csv
import io
import logging
import sqlite3
import threading
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
MAX_STORED_JOBS = 500
_UNFINISHED = (JobStatus.PENDING, JobStatus.RUNNING)
_FINISHED = (JobStatus.COMPLETED, JobStatus.FAILED)
_INTERRUPTED_DETAIL = "interrupted by a service restart"

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
    """Job registry bounded to MAX_STORED_JOBS (finished jobs evicted first, oldest first).

    An in-memory dict serves reads on the hot path; when `db_path` is set the same records
    are mirrored to SQLite so they survive a restart. All SQLite I/O runs off the event
    loop (`asyncio.to_thread`) under a lock, so `create`/`save`/`evict` are async; `get`
    stays sync (cache read). With no `db_path` the store is purely in-memory.
    """

    def __init__(self, db_path: str = "", max_jobs: int = MAX_STORED_JOBS) -> None:
        self._jobs: dict[str, ImportJob] = {}
        self._max_jobs = max_jobs
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._seq = 0

    async def initialize(self) -> None:
        """Open the DB (if configured), load persisted jobs, and reap any left unfinished
        by a process that died — they can never resume, so they become FAILED."""
        if not self._db_path:
            return
        await asyncio.to_thread(self._init_db)
        reaped = sum(1 for j in self._jobs.values() if j.detail == _INTERRUPTED_DETAIL)
        logger.info("JobStore loaded %d jobs from %s (%d reaped)", len(self._jobs), self._db_path, reaped)

    def _init_db(self) -> None:
        with self._lock:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS import_jobs "
                "(job_id TEXT PRIMARY KEY, seq INTEGER, status TEXT, data TEXT)"
            )
            conn.commit()
            for seq, data in conn.execute("SELECT seq, data FROM import_jobs ORDER BY seq").fetchall():
                job = ImportJob.model_validate_json(data)
                if job.status in _UNFINISHED:
                    job.status = JobStatus.FAILED
                    job.detail = _INTERRUPTED_DETAIL
                    conn.execute(
                        "UPDATE import_jobs SET status=?, data=? WHERE job_id=?",
                        (job.status.value, job.model_dump_json(), job.job_id),
                    )
                self._jobs[job.job_id] = job
                self._seq = max(self._seq, seq)
            conn.commit()
            self._conn = conn

    async def create(self, filename: str) -> ImportJob:
        job = ImportJob(job_id=uuid.uuid4().hex, filename=filename)
        self._jobs[job.job_id] = job
        self._seq += 1
        if self._conn is not None:
            await asyncio.to_thread(
                self._exec,
                "INSERT OR REPLACE INTO import_jobs (job_id, seq, status, data) VALUES (?,?,?,?)",
                (job.job_id, self._seq, job.status.value, job.model_dump_json()),
            )
        await self._evict()
        return job

    def get(self, job_id: str) -> ImportJob | None:
        return self._jobs.get(job_id)

    async def save(self, job: ImportJob) -> None:
        """Persist progress / terminal status of an already-created job."""
        self._jobs[job.job_id] = job
        if self._conn is not None:
            await asyncio.to_thread(
                self._exec,
                "UPDATE import_jobs SET status=?, data=? WHERE job_id=?",
                (job.status.value, job.model_dump_json(), job.job_id),
            )

    async def _evict(self) -> None:
        while len(self._jobs) > self._max_jobs:
            # dicts preserve insertion order → the first finished job is the oldest one
            victim = next((jid for jid, j in self._jobs.items() if j.status in _FINISHED), None)
            victim = victim if victim is not None else next(iter(self._jobs))
            del self._jobs[victim]
            if self._conn is not None:
                await asyncio.to_thread(self._exec, "DELETE FROM import_jobs WHERE job_id=?", (victim,))

    def _exec(self, sql: str, params: tuple[Any, ...]) -> None:
        with self._lock:
            assert self._conn is not None
            self._conn.execute(sql, params)
            self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


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


async def _ingest_in_chunks(
    job: ImportJob, products: list[ProductIn], ingest: IngestService, jobs: JobStore
) -> None:
    """Chunk-ingest products, persisting progress after each chunk, and set the terminal
    status. Any ingest error fails the job (recording how far it got) rather than raising."""
    try:
        for start in range(0, len(products), INGEST_CHUNK):
            chunk = products[start : start + INGEST_CHUNK]
            await ingest.upsert_products(chunk)
            job.processed += len(chunk)
            await jobs.save(job)
        job.status = JobStatus.COMPLETED
    except Exception as exc:
        job.status = JobStatus.FAILED
        job.detail = f"Ingestion error after {job.processed} rows: {exc}"
        logger.exception("Job %s failed during ingestion", job.job_id)
    await jobs.save(job)


async def run_import(
    job: ImportJob,
    content: bytes,
    mapping: dict[str, str] | None,
    ingest: IngestService,
    jobs: JobStore,
) -> None:
    job.status = JobStatus.RUNNING
    await jobs.save(job)
    try:
        raw_rows = parse_rows(job.filename, content)
    except Exception as exc:
        job.status = JobStatus.FAILED
        job.detail = f"Cannot parse file: {exc}"
        logger.exception("Import %s failed to parse", job.job_id)
        await jobs.save(job)
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
    await jobs.save(job)
    await _ingest_in_chunks(job, products, ingest, jobs)


async def run_batch(job: ImportJob, products: list[ProductIn], ingest: IngestService, jobs: JobStore) -> None:
    """Background ingest of an already-parsed API batch — the async counterpart of the
    synchronous `:batch` endpoint, so a large or enrichment-heavy batch can't time out
    the request. Progress and terminal status land in the (durable) JobStore."""
    job.status = JobStatus.RUNNING
    job.total = len(products)
    await jobs.save(job)
    await _ingest_in_chunks(job, products, ingest, jobs)
