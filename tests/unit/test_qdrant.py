"""Unit coverage for the Qdrant hybrid query construction (no live server): the request
must carry two prefetch branches (dense + sparse_text) fused by server-side RRF, with the
user filter threaded into both. The client is replaced by a capturing stub."""

from types import SimpleNamespace

from qdrant_client import models

from app.core.config import Settings
from app.services.qdrant import QdrantService, build_filter


class _CaptureClient:
    def __init__(self) -> None:
        self.captured: dict = {}

    async def query_points(self, **kwargs):
        self.captured = kwargs
        return SimpleNamespace(points=[])


def _service_with_capture() -> tuple[QdrantService, _CaptureClient]:
    service = QdrantService(Settings(_env_file=None))
    client = _CaptureClient()
    service.client = client  # type: ignore[assignment]
    return service, client


async def test_hybrid_query_builds_two_prefetch_rrf():
    service, client = _service_with_capture()
    await service.hybrid_query([0.1, 0.2], models.SparseVector(indices=[1], values=[0.5]), None, limit=10)
    cap = client.captured
    prefetch = cap["prefetch"]
    assert len(prefetch) == 2
    assert {p.using for p in prefetch} == {"dense", "sparse_text"}
    assert all(p.limit == 50 for p in prefetch)  # max(prefetch_limit=50, limit=10)
    assert isinstance(cap["query"], models.FusionQuery)
    assert cap["query"].fusion == models.Fusion.RRF
    assert cap["limit"] == 10
    assert cap["with_payload"] is True


async def test_hybrid_query_prefetch_grows_with_limit():
    service, client = _service_with_capture()
    await service.hybrid_query([0.1], models.SparseVector(indices=[], values=[]), None, limit=80)
    prefetch = client.captured["prefetch"]
    assert all(p.limit == 80 for p in prefetch)  # limit exceeds prefetch_limit → use limit


async def test_hybrid_query_threads_filter_into_both_branches():
    service, client = _service_with_capture()
    flt = build_filter(None, include_archived=False)  # non-None: excludes archived
    await service.hybrid_query([0.1], models.SparseVector(indices=[], values=[]), flt, limit=5)
    prefetch = client.captured["prefetch"]
    assert all(p.filter is flt for p in prefetch)
