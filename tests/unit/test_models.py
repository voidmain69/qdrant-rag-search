import pytest
from pydantic import ValidationError

from app.models.product import ProductIn
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


class TestSearchRequest:
    def test_whitespace_only_query_rejected(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="   ")

    def test_query_stripped(self):
        assert SearchRequest(query="  дриль ").query == "дриль"
