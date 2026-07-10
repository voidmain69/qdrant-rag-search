"""Strict search mode end-to-end at the service level (Qdrant and embedder stubbed).

Scenario: "мат плата з hdmi на 1200" — the catalog has LGA 1200 boards, but only one
of them has HDMI. Strict mode must return the HDMI board as a confident item and the
HDMI-less board as an alternative that names the missing term.
"""

import pytest
from qdrant_client import models

from app.core.config import Settings
from app.models.search import SearchMode, SearchRequest
from app.services.code_index import CodeIndex
from app.services.coverage import Requirement
from app.services.query_router import SearchService

MB_WITH_HDMI = {
    "external_id": "mb-1",
    "name": "Материнська плата MSI B460M-A PRO",
    "attributes": {"Сокет": "LGA 1200", "Відеовиходи": "HDMI, DVI-D"},
}

MB_NO_HDMI = {
    "external_id": "mb-2",
    "name": "Материнська плата ASUS PRIME H410M-R",
    "attributes": {"Сокет": "LGA 1200", "Відеовиходи": "D-Sub, DVI-D"},
}

# a vector neighbour that shares almost nothing with the query — noise, not an alternative
KETTLE = {
    "external_id": "kettle-1",
    "name": "Чайник електричний Tefal KI270D30",
    "attributes": {"Потужність": "1200 Вт"},
}


class StubQdrant:
    """Returns a fixed nearest-neighbour list, as a real hybrid query would."""

    def __init__(self, payloads: list[dict]):
        self._points = [
            models.ScoredPoint(id=str(i), version=0, score=1.0 - i * 0.1, payload=p)
            for i, p in enumerate(payloads)
        ]

    async def hybrid_query(self, *args, **kwargs):
        return self._points

    async def retrieve_payloads(self, point_ids):
        return {}


class StubEmbedder:
    async def aembed_query(self, dense_text, sparse_text):
        return [0.0], models.SparseVector(indices=[], values=[])


def make_service(payloads: list[dict]) -> SearchService:
    return SearchService(
        settings=Settings(_env_file=None),
        qdrant=StubQdrant(payloads),  # type: ignore[arg-type]
        embedder=StubEmbedder(),  # type: ignore[arg-type]
        code_index=CodeIndex(),
        reranker=None,
    )


QUERY = "мат плата з hdmi на 1200"


async def test_strict_mode_splits_confident_and_alternatives():
    service = make_service([MB_NO_HDMI, MB_WITH_HDMI])  # nearest-first: the wrong one ranks higher
    resp = await service.search(SearchRequest(query=QUERY, mode=SearchMode.STRICT))

    assert [h.product["external_id"] for h in resp.items] == ["mb-1"]
    assert resp.items[0].match.query_coverage == 1.0
    assert resp.total == 1

    assert [h.product["external_id"] for h in resp.alternatives] == ["mb-2"]
    assert resp.alternatives[0].match.missing_terms == ["hdmi"]


async def test_low_coverage_neighbours_are_not_alternatives():
    # the kettle matches "1200" only (coverage 0.25) — below the alternative threshold
    service = make_service([MB_NO_HDMI, KETTLE])
    resp = await service.search(SearchRequest(query=QUERY, mode=SearchMode.STRICT))
    assert resp.items == []
    assert [h.product["external_id"] for h in resp.alternatives] == ["mb-2"]


async def test_strict_mode_all_alternatives_when_nothing_covers():
    service = make_service([MB_NO_HDMI])
    resp = await service.search(SearchRequest(query=QUERY, mode=SearchMode.STRICT))
    assert resp.items == []
    assert resp.total == 0
    assert len(resp.alternatives) == 1
    assert resp.alternatives[0].match.missing_terms == ["hdmi"]


async def test_relaxed_mode_keeps_single_list_but_annotates():
    service = make_service([MB_NO_HDMI, MB_WITH_HDMI])
    resp = await service.search(SearchRequest(query=QUERY, mode=SearchMode.RELAXED))
    assert resp.alternatives == []
    assert len(resp.items) == 2  # nearest-first order untouched
    by_id = {h.product["external_id"]: h for h in resp.items}
    assert by_id["mb-1"].match.query_coverage == 1.0
    assert by_id["mb-2"].match.missing_terms == ["hdmi"]


async def test_strict_mode_pagination_counts_confident_only():
    service = make_service([MB_WITH_HDMI, MB_NO_HDMI])
    resp = await service.search(SearchRequest(query=QUERY, mode=SearchMode.STRICT, limit=1, offset=1))
    assert resp.total == 1  # one confident hit in total
    assert resp.items == []  # offset=1 is past it
    assert len(resp.alternatives) == 1


@pytest.mark.parametrize("mode", [SearchMode.RELAXED, SearchMode.STRICT])
async def test_mode_accepted_in_request_model(mode):
    assert SearchRequest(query="x", mode=mode).mode is mode


class StubUnderstanding:
    """Stands in for the Ollama-backed QueryUnderstandingService."""

    def __init__(self, requirements):
        self.requirements = requirements
        self.calls = 0

    async def extract(self, query):
        self.calls += 1
        return self.requirements


BRUSHLESS_DRILL = {
    "external_id": "drl-1",
    "name": "Шуруповерт акумуляторний Makita DDF484Z",
    "attributes": {"Тип двигуна": "Brushless"},
}

BRUSHLESS_QUERY = "безщітковий шуруповерт makita"

BRUSHLESS_REQS = [
    Requirement("шуруповерт", ("шуруповерт", "дриль-шуруповерт", "screwdriver")),
    Requirement("безщітковий", ("безщітковий", "brushless", "бесщеточный")),
    Requirement("makita", ("makita", "макіта")),
]


async def test_llm_synonyms_make_hit_confident():
    """Lexically "безщітковий" is nowhere in the product; the LLM variant "brushless"
    covers it, so strict mode returns the product as a confident item."""
    service = make_service([BRUSHLESS_DRILL])
    service.understanding = StubUnderstanding(BRUSHLESS_REQS)  # type: ignore[assignment]
    resp = await service.search(SearchRequest(query=BRUSHLESS_QUERY, mode=SearchMode.STRICT))
    assert [h.product["external_id"] for h in resp.items] == ["drl-1"]
    assert resp.items[0].match.query_coverage == 1.0


async def test_without_llm_same_query_degrades_to_alternative():
    service = make_service([BRUSHLESS_DRILL])  # understanding=None → token fallback
    resp = await service.search(SearchRequest(query=BRUSHLESS_QUERY, mode=SearchMode.STRICT))
    assert resp.items == []
    assert [h.product["external_id"] for h in resp.alternatives] == ["drl-1"]
    assert resp.alternatives[0].match.missing_terms == ["безщітковий"]


async def test_llm_failure_falls_back_to_tokens():
    class FailingUnderstanding:
        async def extract(self, query):
            return None  # Ollama down / unparseable

    service = make_service([BRUSHLESS_DRILL])
    service.understanding = FailingUnderstanding()  # type: ignore[assignment]
    resp = await service.search(SearchRequest(query=BRUSHLESS_QUERY, mode=SearchMode.STRICT))
    assert [h.product["external_id"] for h in resp.alternatives] == ["drl-1"]


async def test_relaxed_mode_never_calls_llm():
    service = make_service([BRUSHLESS_DRILL])
    stub = StubUnderstanding(BRUSHLESS_REQS)
    service.understanding = stub  # type: ignore[assignment]
    await service.search(SearchRequest(query=BRUSHLESS_QUERY, mode=SearchMode.RELAXED))
    assert stub.calls == 0


async def test_code_only_strict_skips_understanding():
    # an unambiguous code lookup short-circuits before any hybrid hits exist, so query
    # understanding (a wasted LLM call in strict mode) must not run
    service = make_service([])
    service.code_index.add_product("p1", {"article": "GSB-13-RE"})
    stub = StubUnderstanding(BRUSHLESS_REQS)
    service.understanding = stub  # type: ignore[assignment]
    await service.search(SearchRequest(query="GSB-13-RE", mode=SearchMode.STRICT))
    assert stub.calls == 0


async def test_monitor_unit_spec_is_confident_in_strict_mode():
    # regression for the "27 дюймів 165 гц" class: the correct monitor must be a confident
    # item, not an alternative, even though the query units differ in script from the spec
    monitor = {
        "external_id": "mon-1",
        "name": 'Монітор Samsung Odyssey G5 27"',
        "attributes": {"Діагональ": "27 inch", "Частота оновлення": "165 Hz"},
    }
    service = make_service([monitor])
    resp = await service.search(SearchRequest(query="монітор 27 дюймів 165 гц", mode=SearchMode.STRICT))
    assert [h.product["external_id"] for h in resp.items] == ["mon-1"]
    assert resp.items[0].match.query_coverage == 1.0
