import pytest
from pydantic import ValidationError

from app.models.product import PriceUpdate, ProductIn, ReconcileRequest
from app.models.search import SearchRequest


class TestProductIn:
    def test_external_id_stripped(self):
        p = ProductIn(external_id="  tool-1 ", name=" Дриль  ударний ")
        assert p.external_id == "tool-1"
        assert p.name == "Дриль ударний"

    def test_ean_digits_only(self):
        p = ProductIn(external_id="x", name="y", ean13=" 4006381-333931 ")
        assert p.ean13 == "4006381333931"

    def test_whitespace_only_external_id_rejected(self):
        with pytest.raises(ValidationError):
            ProductIn(external_id="   ", name="y")


class TestPriceUpdate:
    def test_empty_update_rejected(self):
        with pytest.raises(ValidationError):
            PriceUpdate(external_id="x")

    def test_changed_fields_only_includes_provided(self):
        assert PriceUpdate(external_id="x", price=99.0).changed_fields() == {"price": 99.0}
        assert PriceUpdate(external_id="x", in_stock=False).changed_fields() == {"in_stock": False}

    def test_all_fields(self):
        u = PriceUpdate(external_id="x", price=10.5, currency="USD", in_stock=True)
        assert u.changed_fields() == {"price": 10.5, "currency": "USD", "in_stock": True}

    def test_negative_price_rejected(self):
        with pytest.raises(ValidationError):
            PriceUpdate(external_id="x", price=-1)

    def test_stock_false_is_a_valid_change(self):
        # in_stock=False must not be mistaken for "field absent"
        assert PriceUpdate(external_id="x", in_stock=False).changed_fields() == {"in_stock": False}


class TestReconcileRequest:
    def test_dry_run_defaults_true(self):
        # safety default: reconcile never mutates unless explicitly told to
        assert ReconcileRequest(external_ids=["a"]).dry_run is True

    def test_empty_ids_rejected(self):
        with pytest.raises(ValidationError):
            ReconcileRequest(external_ids=[])

    def test_negative_cap_rejected(self):
        with pytest.raises(ValidationError):
            ReconcileRequest(external_ids=["a"], max_archived=-1)


class TestSearchRequest:
    def test_whitespace_only_query_rejected(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="   ")

    def test_query_stripped(self):
        assert SearchRequest(query="  дриль ").query == "дриль"

    def test_offset_upper_bound_rejected(self):
        # deep pagination is unsupported and an unbounded offset is a fetch DoS lever
        SearchRequest(query="x", offset=1000)  # at the cap: allowed
        with pytest.raises(ValidationError):
            SearchRequest(query="x", offset=1001)
