"""SearchService orchestration at the service level (Qdrant + embedder stubbed):
the relaxed hybrid path, code-branch short-circuit, and the merge/pinning rules that
otherwise only run under the integration suite."""

import pytest
from fastapi import HTTPException
from qdrant_client import models

from app.core.config import Settings
from app.models.search import MatchBranch, SearchRequest
from app.services.code_index import CodeIndex
from app.services.query_router import SearchService


class StubEmbedder:
    async def aembed_query(self, dense_text, sparse_text):
        return [0.0], models.SparseVector(indices=[], values=[])


class StubQdrant:
    """Hybrid results as ScoredPoints (nearest-first); retrieve_payloads serves the
    code branch. Counts hybrid_query calls so a short-circuit can be asserted."""

    def __init__(self, hybrid=None, stored=None):
        self._points = [
            models.ScoredPoint(id=str(i), version=0, score=score, payload=payload)
            for i, (payload, score) in enumerate(hybrid or [])
        ]
        self._stored = stored or {}
        self.hybrid_calls = 0

    async def hybrid_query(self, dense, sparse, flt, limit):
        self.hybrid_calls += 1
        return self._points

    async def retrieve_payloads(self, point_ids):
        return {pid: self._stored[pid] for pid in point_ids if pid in self._stored}


def make_service(hybrid=None, stored=None, codes=None):
    code_index = CodeIndex()
    for pid, fields in (codes or {}).items():
        code_index.add_product(pid, fields)
    qdrant = StubQdrant(hybrid, stored)
    service = SearchService(
        settings=Settings(_env_file=None),
        qdrant=qdrant,  # type: ignore[arg-type]
        embedder=StubEmbedder(),  # type: ignore[arg-type]
        code_index=code_index,
        reranker=None,
    )
    return service, qdrant


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


async def test_relaxed_text_returns_hybrid_nearest_first():
    hybrid = [
        ({"external_id": "a", "name": "Дриль ударний A"}, 0.03),
        ({"external_id": "b", "name": "Дриль ударний B"}, 0.02),
        ({"external_id": "c", "name": "Дриль ударний C"}, 0.01),
    ]
    service, qdrant = make_service(hybrid=hybrid)
    resp = await service.search(SearchRequest(query="дриль ударний"))
    assert [h.product["external_id"] for h in resp.items] == ["a", "b", "c"]
    assert all(h.match.branch is MatchBranch.HYBRID for h in resp.items)
    assert resp.alternatives == []
    assert qdrant.hybrid_calls == 1


async def test_code_only_strong_hit_skips_vector_search():
    stored = {"p1": {"external_id": "p1", "name": "Дриль Bosch", "article": "GSB-13-RE"}}
    service, qdrant = make_service(stored=stored, codes={"p1": {"article": "GSB-13-RE"}})
    resp = await service.search(SearchRequest(query="GSB-13-RE"))
    assert qdrant.hybrid_calls == 0  # unambiguous code lookup short-circuits
    assert [h.product["external_id"] for h in resp.items] == ["p1"]
    assert resp.items[0].match.branch is MatchBranch.EXACT
    assert resp.items[0].score == 1.0


async def test_mixed_pins_strong_code_above_hybrid():
    stored = {"p1": {"external_id": "p1", "name": "Дриль Bosch GSB-13-RE", "article": "GSB-13-RE"}}
    hybrid = [({"external_id": "x", "name": "Шуруповерт Makita"}, 0.02)]
    service, qdrant = make_service(hybrid=hybrid, stored=stored, codes={"p1": {"article": "GSB-13-RE"}})
    resp = await service.search(SearchRequest(query="дриль GSB-13-RE"))
    assert qdrant.hybrid_calls == 1  # mixed runs both branches
    assert [h.product["external_id"] for h in resp.items] == ["p1", "x"]
    assert resp.items[0].match.branch is MatchBranch.EXACT
    assert resp.items[1].match.branch is MatchBranch.HYBRID


async def test_merge_dedups_by_external_id():
    stored = {"p1": {"external_id": "p1", "name": "Дриль GSB-13-RE", "article": "GSB-13-RE"}}
    # the hybrid branch also surfaces p1 — it must not appear twice; the pinned code hit wins
    hybrid = [({"external_id": "p1", "name": "Дриль GSB-13-RE"}, 0.02)]
    service, _ = make_service(hybrid=hybrid, stored=stored, codes={"p1": {"article": "GSB-13-RE"}})
    resp = await service.search(SearchRequest(query="дриль GSB-13-RE"))
    assert [h.product["external_id"] for h in resp.items] == ["p1"]
    assert resp.items[0].match.branch is MatchBranch.EXACT


async def test_offset_and_limit_slice_relaxed():
    hybrid = [
        ({"external_id": e, "name": f"Дриль {e}"}, s) for e, s in [("a", 0.03), ("b", 0.02), ("c", 0.01)]
    ]
    service, _ = make_service(hybrid=hybrid)
    resp = await service.search(SearchRequest(query="дриль", limit=1, offset=1))
    assert resp.total == 3  # counts the whole fetched window, not just the page
    assert [h.product["external_id"] for h in resp.items] == ["b"]
