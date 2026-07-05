"""End-to-end tests against a live docker compose stack.

Run:
    docker compose up -d --build     # wait for healthy
    uv run pytest -m integration -q

Env overrides: E2E_BASE_URL (default http://localhost:8000), E2E_API_KEY.
"""

import json
import os
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.integration

BASE_URL = os.environ.get("E2E_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("E2E_API_KEY", "change-me-secret-key")


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=BASE_URL, headers={"X-API-Key": API_KEY}, timeout=120) as c:
        ready = c.get("/ready")
        if ready.status_code != 200:
            pytest.skip(f"stack not ready at {BASE_URL}: {ready.status_code}")
        yield c


@pytest.fixture(scope="module")
def ingested(client):
    data = json.loads(
        (Path(__file__).resolve().parents[2] / "data" / "sample_products.json").read_text("utf-8")
    )
    resp = client.post("/api/v1/products:batch", json={"items": data})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["failed"] == 0
    return body


def search(client, query: str, **kwargs) -> dict:
    resp = client.post("/api/v1/search", json={"query": query, "limit": 5, **kwargs})
    assert resp.status_code == 200, resp.text
    return resp.json()


class TestAuth:
    def test_missing_key_rejected(self):
        with httpx.Client(base_url=BASE_URL, timeout=30) as anon:
            resp = anon.post("/api/v1/search", json={"query": "test"})
            assert resp.status_code == 401

    def test_health_open(self):
        with httpx.Client(base_url=BASE_URL, timeout=30) as anon:
            assert anon.get("/health").status_code == 200


class TestSearchBranches:
    def test_semantic_phrase(self, client, ingested):
        data = search(client, "бездротовий пилосос для дому")
        assert data["query_kind"] == "text"
        names = [h["product"]["name"] for h in data["items"]]
        assert any("Dyson" in n or "пилосос" in n.lower() for n in names[:2])
        assert data["items"][0]["match"]["branch"] == "hybrid"

    def test_characteristics_hybrid_not_code(self, client, ingested):
        data = search(client, "акумулятор 18V 5Ah Li-Ion")
        assert data["query_kind"] == "text"
        assert data["items"][0]["product"]["external_id"] == "tool-004"

    def test_exact_article(self, client, ingested):
        data = search(client, "GSB-13-RE")
        assert data["query_kind"] == "code_only"
        top = data["items"][0]
        assert top["product"]["external_id"] == "tool-001"
        assert top["match"]["branch"] == "exact"

    def test_article_typo_fuzzy(self, client, ingested):
        data = search(client, "GSB-13-RF")  # E -> F typo
        top = data["items"][0]
        assert top["product"]["external_id"] == "tool-001"
        assert top["match"]["branch"] == "fuzzy"

    def test_exact_ean(self, client, ingested):
        data = search(client, "4006381333931")
        top = data["items"][0]
        assert top["product"]["external_id"] == "tool-001"
        assert top["match"]["branch"] == "exact"

    def test_ean_wrong_digit_corrected(self, client, ingested):
        data = search(client, "4006381333932")  # broken check digit
        top = data["items"][0]
        assert top["product"]["external_id"] == "tool-001"
        assert top["match"]["branch"] == "ean_corrected"

    def test_mixed_query_pins_code_hit(self, client, ingested):
        data = search(client, "дриль GSB13RE з кейсом")
        assert data["query_kind"] == "mixed"
        top = data["items"][0]
        assert top["product"]["external_id"] == "tool-001"
        assert top["match"]["branch"] == "exact"

    def test_filters(self, client, ingested):
        data = search(client, "пилосос", filters={"brand": "Samsung"})
        assert data["items"]
        for hit in data["items"]:
            assert hit["product"]["brand"] == "Samsung"


class TestStrictMode:
    """No motherboard in the catalog has both HDMI and socket 1200 except mb-1:
    strict mode must separate the confident hit from the near-miss alternative."""

    MOTHERBOARDS = (
        {
            "external_id": "e2e-mb-1",
            "name": "Материнська плата MSI B460M-A PRO",
            "brand": "MSI",
            "category": "Материнські плати",
            "attributes": {"Сокет": "LGA 1200", "Відеовиходи": "HDMI, DVI-D"},
        },
        {
            "external_id": "e2e-mb-2",
            "name": "Материнська плата ASUS PRIME H410M-R",
            "brand": "ASUS",
            "category": "Материнські плати",
            "attributes": {"Сокет": "LGA 1200", "Відеовиходи": "D-Sub, DVI-D"},
        },
    )

    @pytest.fixture(scope="class")
    def boards(self, client, ingested):
        resp = client.post("/api/v1/products:batch", json={"items": self.MOTHERBOARDS})
        assert resp.status_code == 200, resp.text
        assert resp.json()["failed"] == 0

    def test_strict_splits_confident_from_alternatives(self, client, boards):
        data = search(client, "мат плата з hdmi на 1200", mode="strict")
        item_ids = [h["product"]["external_id"] for h in data["items"]]
        alt_ids = [h["product"]["external_id"] for h in data["alternatives"]]

        assert "e2e-mb-1" in item_ids
        assert all(h["match"]["query_coverage"] == 1.0 for h in data["items"])

        assert "e2e-mb-2" in alt_ids
        near_miss = next(h for h in data["alternatives"] if h["product"]["external_id"] == "e2e-mb-2")
        assert near_miss["match"]["missing_terms"] == ["hdmi"]

    def test_relaxed_annotates_but_does_not_split(self, client, boards):
        data = search(client, "мат плата з hdmi на 1200")
        assert data["alternatives"] == []
        hits = {h["product"]["external_id"]: h for h in data["items"]}
        assert hits["e2e-mb-1"]["match"]["query_coverage"] == 1.0
        assert hits["e2e-mb-2"]["match"]["missing_terms"] == ["hdmi"]


class TestLifecycle:
    def test_update_reflects_in_search(self, client, ingested):
        product = {
            "external_id": "e2e-tmp-1",
            "name": "Тестовий генератор інверторний 2 кВт",
            "brand": "TestBrand",
            "category": "Генератори",
            "article": "E2E-GEN-2000",
        }
        resp = client.post("/api/v1/products", json=product)
        assert resp.status_code == 200
        data = search(client, "E2E-GEN-2000")
        assert data["items"][0]["product"]["external_id"] == "e2e-tmp-1"

        product["name"] = "Тестовий генератор інверторний 2 кВт ОНОВЛЕНО"
        resp = client.put("/api/v1/products/e2e-tmp-1", json=product)
        assert resp.status_code == 200
        data = search(client, "E2E-GEN-2000")
        assert "ОНОВЛЕНО" in data["items"][0]["product"]["name"]

    def test_idempotent_reingest_keeps_count(self, client, ingested):
        data = json.loads(
            (Path(__file__).resolve().parents[2] / "data" / "sample_products.json").read_text("utf-8")
        )
        resp = client.post("/api/v1/products:batch", json={"items": data})
        assert resp.status_code == 200
        # same external_ids -> same point ids -> re-searching the same article still 1 result pinned
        found = search(client, "GSB-13-RE")
        top_ids = [h["product"]["external_id"] for h in found["items"]]
        assert top_ids.count("tool-001") == 1

    def test_delete(self, client, ingested):
        resp = client.delete("/api/v1/products/e2e-tmp-1")
        assert resp.status_code == 204
        assert client.delete("/api/v1/products/e2e-tmp-1").status_code == 404
        data = search(client, "E2E-GEN-2000")
        assert all(h["product"]["external_id"] != "e2e-tmp-1" for h in data["items"])
