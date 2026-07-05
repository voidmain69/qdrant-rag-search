from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RowError(BaseModel):
    row: int
    error: str


class ImportJob(BaseModel):
    job_id: str
    filename: str
    status: JobStatus = JobStatus.PENDING
    total: int = 0
    processed: int = 0
    failed: int = 0
    errors: list[RowError] = Field(default_factory=list, description="First 100 row-level errors")
    detail: str | None = None
