from app.models.product import ProductIn
from app.services.normalization import (
    compose_dense_text,
    compose_sparse_text,
    is_code_like,
    is_ean_like,
    norm_code,
    skeleton,
)


class TestNormCode:
    def test_strips_separators_and_uppercases(self):
        assert norm_code("gsb 13-re") == "GSB13RE"
        assert norm_code("ABC_12.34/56") == "ABC123456"

    def test_cyrillic_homoglyphs_mapped_to_latin(self):
        # Cyrillic Н, В look identical to Latin H, B
        assert norm_code("НВ-1234") == "HB1234"
        assert norm_code("СОРТ-500") == "COPT500"

    def test_nfkc_normalization(self):
        assert norm_code("ＧＳＢ１３") == "GSB13"  # fullwidth forms

    def test_non_homoglyph_cyrillic_survives(self):
        # Д and Ж have no Latin twin — they must survive as-is
        assert norm_code("ДЖ-100") == "ДЖ100"


class TestSkeleton:
    def test_ocr_folding(self):
        assert skeleton("BOSCH06") == skeleton("BQSCH-O6".replace("-", ""))
        assert skeleton("O0I1L") == "00111"
        assert skeleton("GSB") == "658"

    def test_cyrillic_ze_folds_to_three(self):
        # Cyrillic З ~ digit 3, Latin B ~ 8: both spellings collapse to one skeleton
        assert skeleton(norm_code("ЗЕВ-1")) == skeleton(norm_code("3EB-1"))


class TestIsCodeLike:
    def test_classic_skus(self):
        assert is_code_like("GSB-13-RE")
        assert is_code_like("GSB13RE")
        assert is_code_like("HW-1234K")
        assert is_code_like("4006381333931")
        assert is_code_like("A123456")

    def test_units_are_not_codes(self):
        for token in ["18V", "5Ah", "230V", "1500W", "500ГБ", "18в", "10мм", "0.5л", "2000mAh"]:
            assert not is_code_like(token), token

    def test_words_are_not_codes(self):
        for token in ["пилосос", "бездротовий", "дриль", "Bosch", "Li-Ion", "АКУМУЛЯТОР"]:
            assert not is_code_like(token), token

    def test_short_tokens_rejected(self):
        assert not is_code_like("X1")
        assert not is_code_like("13")

    def test_cyrillic_homoglyph_code(self):
        assert is_code_like("НВ-1234")  # typed on a Ukrainian keyboard


class TestIsEanLike:
    def test_ean_lengths(self):
        assert is_ean_like("4006381333931")
        assert is_ean_like("400638133393")  # 12
        assert is_ean_like("14006381333938")  # 14
        assert not is_ean_like("12345")
        assert not is_ean_like("GSB13RE")


class TestTextComposition:
    def _product(self) -> ProductIn:
        return ProductIn(
            external_id="p1",
            name="Дриль ударний Bosch GSB 13 RE",
            brand="Bosch",
            category="Електроінструмент",
            article="GSB-13-RE",
            product_code="060114E600",
            ean13="3165140371940",
            attributes={"Потужність": "600 Вт", "Патрон": "ШЗП 13 мм"},
            description="Компактний ударний дриль для дому.",
        )

    def test_dense_text_excludes_codes(self):
        text = compose_dense_text(self._product())
        assert "Дриль ударний" in text
        assert "Бренд: Bosch" in text
        assert "Потужність: 600 Вт" in text
        assert "GSB-13-RE" not in text
        assert "3165140371940" not in text
        assert "060114E600" not in text

    def test_sparse_text_includes_raw_and_norm_codes(self):
        text = compose_sparse_text(self._product())
        assert "GSB-13-RE" in text
        assert "GSB13RE" in text
        assert "3165140371940" in text
        assert "060114E600" in text
