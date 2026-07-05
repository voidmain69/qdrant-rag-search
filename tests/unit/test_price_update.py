"""Price / availability updates are payload-only: no embedding, no CodeIndex change,
unknown ids reported (not fatal). The embedder and code index are deliberately None to
prove they are never touched."""

from app.core.config import Settings
from app.models.product import PriceUpdate
from app.services.ingest import IngestService
from app.utils.ids import point_id_for


class StubQdrant:
    def __init__(self, existing_external_ids: list[str]):
        self._existing = {point_id_for(eid) for eid in existing_external_ids}
        self.set_calls: list[tuple[str, dict]] = []

    async def retrieve_existing(self, point_ids: list[str]) -> set[str]:
        return {pid for pid in point_ids if pid in self._existing}

    async def set_payload(self, point_id: str, payload: dict) -> None:
        self.set_calls.append((point_id, payload))


def make_service(existing: list[str]) -> tuple[IngestService, StubQdrant]:
    qdrant = StubQdrant(existing)
    service = IngestService(
        settings=Settings(_env_file=None),
        embedder=None,  # type: ignore[arg-type]  — must never be used
        qdrant=qdrant,  # type: ignore[arg-type]
        code_index=None,  # type: ignore[arg-type]  — must never be used
    )
    return service, qdrant


async def test_updates_only_changed_fields():
    service, qdrant = make_service(["p1"])
    result = await service.update_prices([PriceUpdate(external_id="p1", price=42.0, in_stock=False)])

    assert result.succeeded == 1 and result.failed == 0
    assert len(qdrant.set_calls) == 1
    point_id, payload = qdrant.set_calls[0]
    assert point_id == point_id_for("p1")
    assert payload["price"] == 42.0
    assert payload["in_stock"] is False
    assert "updated_at" in payload
    # untouched fields are not part of the patch
    assert "currency" not in payload
    assert "name" not in payload


async def test_unknown_id_is_reported_not_fatal():
    service, qdrant = make_service(["p1"])
    result = await service.update_prices(
        [
            PriceUpdate(external_id="p1", price=1.0),
            PriceUpdate(external_id="ghost", price=2.0),
        ]
    )
    assert result.total == 2
    assert result.succeeded == 1
    assert result.failed == 1
    ghost = next(i for i in result.items if i.external_id == "ghost")
    assert ghost.ok is False and ghost.error == "Product not found"
    # only the existing product was written
    assert [c[0] for c in qdrant.set_calls] == [point_id_for("p1")]


async def test_duplicate_ids_last_write_wins():
    service, qdrant = make_service(["p1"])
    result = await service.update_prices(
        [PriceUpdate(external_id="p1", price=1.0), PriceUpdate(external_id="p1", price=9.0)]
    )
    assert result.total == 1
    assert len(qdrant.set_calls) == 1
    assert qdrant.set_calls[0][1]["price"] == 9.0
