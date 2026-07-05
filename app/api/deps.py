from fastapi import Request

from app.services.importer import JobStore
from app.services.ingest import IngestService
from app.services.query_router import SearchService


def get_search_service(request: Request) -> SearchService:
    return request.app.state.search_service


def get_ingest_service(request: Request) -> IngestService:
    return request.app.state.ingest_service


def get_job_store(request: Request) -> JobStore:
    return request.app.state.job_store
