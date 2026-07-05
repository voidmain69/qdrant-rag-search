from app.models.search import QueryKind
from app.services.query_router import classify


class TestClassify:
    def test_natural_language_ukrainian(self):
        cls = classify("бездротовий пилосос для дому")
        assert cls.kind == QueryKind.TEXT
        assert cls.code_tokens == []

    def test_characteristics_with_units_stay_text(self):
        cls = classify("акумулятор 18V 5Ah Li-Ion")
        assert cls.kind == QueryKind.TEXT

    def test_pure_article(self):
        cls = classify("GSB-13-RE")
        assert cls.kind == QueryKind.CODE_ONLY
        assert cls.code_tokens == ["GSB-13-RE"]

    def test_pure_ean(self):
        cls = classify("4006381333931")
        assert cls.kind == QueryKind.CODE_ONLY

    def test_mixed_query(self):
        cls = classify("дриль GSB13RE з кейсом")
        assert cls.kind == QueryKind.MIXED
        assert cls.code_tokens == ["GSB13RE"]
        assert "дриль" in cls.text and "кейсом" in cls.text

    def test_mixed_with_ean(self):
        cls = classify("фільтр 4006381333931")
        assert cls.kind == QueryKind.MIXED
        assert cls.code_tokens == ["4006381333931"]

    def test_brand_name_not_code(self):
        cls = classify("перфоратор Makita")
        assert cls.kind == QueryKind.TEXT

    def test_multiple_codes(self):
        cls = classify("GSB-13-RE HW-1234K")
        assert cls.kind == QueryKind.CODE_ONLY
        assert len(cls.code_tokens) == 2
