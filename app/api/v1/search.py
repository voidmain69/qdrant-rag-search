from fastapi import APIRouter, Depends

from app.api.deps import get_search_service
from app.models.search import SearchRequest, SearchResponse
from app.services.query_router import SearchService

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(
    body: SearchRequest, service: SearchService = Depends(get_search_service)
) -> SearchResponse:
    return await service.search(body)
