"""Catalog sync: content-hash skip (no needless re-embedding), archive/restore,
bulk delete, snapshot reconcile (archives orphans, never deletes) with dry-run + cap,
diff and stats. Qdrant and the embedder are stubbed so we can assert exactly when
embedding does and does not happen."""

import pytest
from fastapi import HTTPException
from qdrant_client import models

from app.core.config import Settings
from app.models.product import ProductIn
from app.services.code_index import CodeIndex
from app.services.ingest import IngestService
from app.utils.ids import point_id_for


class StubQdrant:
    def __init__(self) -> None:
        self.store: dict[str, dict] = {}  # point_id -> payload
        self.embed_upserts = 0
        self.set_payload_calls: list[tuple[str, dict]] = []
        self.deleted: list[str] = []

    async def retrieve_payloads(self, point_ids):
        return {pid: dict(self.store[pid]) for pid in point_ids if pid in self.store}

    async def retrieve_existing(self, point_ids):
        return {pid for pid in point_ids if pid in self.store}

    async def upsert_points(self, points):
        self.embed_upserts += 1
        for pt in points:
            self.store[pt.id] = dict(pt.payload)

    async def set_payload(self, point_id, payload):
        self.set_payload_calls.append((point_id, dict(payload)))
        if point_id in self.store:
            self.store[point_id].update(payload)

    async def delete_points(self, point_ids):
        for pid in point_ids:
            self.deleted.append(pid)
            self.store.pop(pid, None)

    async def all_external_ids_status(self):
        return [(p["external_id"], p.get("status")) for p in self.store.values() if p.get("external_id")]

    async def count(self, count_filter=None):
        if count_filter is None:
            return len(self.store)
        return sum(1 for p in self.store.values() if p.get("status") == "archived")


class StubEmbedder:
    def __init__(self) -> None:
        self.embed_calls = 0

    async def aembed_docs(self, dense_texts, sparse_texts):
        self.embed_calls += 1
        n = len(dense_texts)
        dense = [[0.0] for _ in range(n)]
        sparse = [models.SparseVector(indices=[], values=[]) for _ in range(n)]
        return dense, sparse


def make_service(**settings_kwargs):
    qdrant = StubQdrant()
    embedder = StubEmbedder()
    service = IngestService(
        settings=Settings(_env_file=None, **settings_kwargs),
        embedder=embedder,  # type: ignore[arg-type]
        qdrant=qdrant,  # type: ignore[arg-type]
        code_index=CodeIndex(),
    )
    return service, qdrant, embedder


def product(external_id: str, **kw) -> ProductIn:
    return ProductIn(external_id=external_id, name=kw.pop("name", f"Товар {external_id}"), **kw)


class TestContentHashSkip:
    async def test_new_product_is_embedded(self):
        service, qdrant, embedder = make_service()
        await service.upsert_products([product("p1", price=100.0)])
        assert embedder.embed_calls == 1
        assert point_id_for("p1") in qdrant.store

    async def test_unchanged_repush_skips_embedding(self):
        service, qdrant, embedder = make_service()
        await service.upsert_products([product("p1", price=100.0)])
        await service.upsert_products([product("p1", price=100.0)])  # identical content
        assert embedder.embed_calls == 1  # not re-embedded
        # the second push still refreshed the payload (price/status/updated_at)
        assert any(pid == point_id_for("p1") for pid, _ in qdrant.set_payload_calls)

    async def test_price_only_change_skips_embedding(self):
        service, qdrant, embedder = make_service()
        await service.upsert_products([product("p1", price=100.0)])
        await service.upsert_products([product("p1", price=79.0)])  # only price differs
        assert embedder.embed_calls == 1
        assert qdrant.store[point_id_for("p1")]["price"] == 79.0

    async def test_content_change_reembeds(self):
        service, _q, embedder = make_service()
        await service.upsert_products([product("p1", name="Стара назва")])
        await service.upsert_products([product("p1", name="Нова назва")])
        assert embedder.embed_calls == 2

    async def test_reembed_unchanged_setting_forces_embedding(self):
        service, _q, embedder = make_service(reembed_unchanged=True)
        await service.upsert_products([product("p1")])
        await service.upsert_products([product("p1")])
        assert embedder.embed_calls == 2


class TestArchive:
    async def test_archive_and_restore(self):
        service, qdrant, _e = make_service()
        await service.upsert_products([product("p1")])
        r = await service.set_archived(["p1"], archived=True)
        assert r.succeeded == 1
        assert qdrant.store[point_id_for("p1")]["status"] == "archived"
        await service.set_archived(["p1"], archived=False)
        assert qdrant.store[point_id_for("p1")]["status"] == "active"

    async def test_unknown_id_reported(self):
        service, _q, _e = make_service()
        r = await service.set_archived(["ghost"], archived=True)
        assert r.failed == 1 and r.items[0].error == "Product not found"


class TestBulkDelete:
    async def test_delete_removes_and_reports(self):
        service, qdrant, _e = make_service()
        await service.upsert_products([product("p1"), product("p2")])
        r = await service.delete_products(["p1", "ghost"])
        assert r.succeeded == 1 and r.failed == 1
        assert point_id_for("p1") in qdrant.deleted
        assert point_id_for("p1") not in qdrant.store
        assert point_id_for("p2") in qdrant.store  # untouched


class TestReconcile:
    async def _seed(self):
        service, qdrant, _e = make_service()
        await service.upsert_products([product("p1"), product("p2"), product("p3")])
        return service, qdrant

    async def test_dry_run_reports_without_mutating(self):
        service, qdrant = await self._seed()
        r = await service.reconcile(["p1"], dry_run=True, max_archived=None)
        assert r.dry_run is True
        assert r.archived_count == 2
        assert set(r.external_ids) == {"p2", "p3"}
        assert all(p["status"] == "active" for p in qdrant.store.values())  # unchanged

    async def test_applies_archives_orphans_not_deletes(self):
        service, qdrant = await self._seed()
        r = await service.reconcile(["p1"], dry_run=False, max_archived=None)
        assert r.archived_count == 2
        assert len(qdrant.store) == 3  # nothing deleted
        assert qdrant.store[point_id_for("p1")]["status"] == "active"
        assert qdrant.store[point_id_for("p2")]["status"] == "archived"
        assert qdrant.store[point_id_for("p3")]["status"] == "archived"

    async def test_already_archived_not_recounted(self):
        service, qdrant = await self._seed()
        await service.set_archived(["p2"], archived=True)
        r = await service.reconcile(["p1"], dry_run=False, max_archived=None)
        assert r.external_ids == ["p3"]  # p2 was already archived

    async def test_cap_refuses(self):
        service, _q = await self._seed()
        with pytest.raises(HTTPException) as exc:
            await service.reconcile(["p1"], dry_run=False, max_archived=1)  # 2 orphans > 1
        assert exc.value.status_code == 400


class TestDiffAndStats:
    async def test_diff(self):
        service, _q, _e = make_service()
        await service.upsert_products([product("p1"), product("p2")])
        r = await service.diff(["p1", "p3"])
        assert r.missing_in_index == ["p3"]  # source has p3, index doesn't
        assert r.extra_in_index == ["p2"]  # index has p2, source doesn't

    async def test_stats(self):
        service, _q, _e = make_service()
        await service.upsert_products([product("p1"), product("p2")])
        await service.set_archived(["p2"], archived=True)
        r = await service.stats()
        assert r.total == 2 and r.active == 1 and r.archived == 1
