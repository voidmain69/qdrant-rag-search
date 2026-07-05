import orjson
from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, UploadFile

from app.api.deps import get_ingest_service, get_job_store
from app.models.imports import ImportJob
from app.services.importer import JobStore, run_import
from app.services.ingest import IngestService

router = APIRouter(tags=["imports"])

MAX_UPLOAD_BYTES = 100 * 1024 * 1024


@router.post("/imports", response_model=ImportJob, status_code=202)
async def start_import(
    background: BackgroundTasks,
    file: UploadFile,
    column_mapping: str | None = Form(default=None, description='JSON: {"source column": "field"}'),
    jobs: JobStore = Depends(get_job_store),
    ingest: IngestService = Depends(get_ingest_service),
) -> ImportJob:
    mapping: dict[str, str] | None = None
    if column_mapping:
        try:
            mapping = orjson.loads(column_mapping)
        except orjson.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail=f"column_mapping is not valid JSON: {exc}") from exc
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 100 MB)")
    job = jobs.create(file.filename or "upload")
    background.add_task(run_import, job, content, mapping, ingest)
    return job


@router.get("/imports/{job_id}", response_model=ImportJob)
async def get_import(job_id: str, jobs: JobStore = Depends(get_job_store)) -> ImportJob:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Import job '{job_id}' not found")
    return job
