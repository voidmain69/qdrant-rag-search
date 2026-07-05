import pytest
from fastapi import HTTPException

from app.models.search import SearchRequest
from app.services.query_router import SearchService


async def test_rerank_disabled_returns_400_for_every_query_kind():
    """rerank=true on an instance without a reranker must fail up-front — including
    code_only queries, which never reach the hybrid branch."""
    service = SearchService(
        settings=None,  # type: ignore[arg-type]
        qdrant=None,  # type: ignore[arg-type]
        embedder=None,  # type: ignore[arg-type]
        code_index=None,  # type: ignore[arg-type]
        reranker=None,
    )
    for query in ["GSB-13-RE", "дриль ударний", "дриль GSB13RE"]:
        with pytest.raises(HTTPException) as exc_info:
            await service.search(SearchRequest(query=query, rerank=True))
        assert exc_info.value.status_code == 400
