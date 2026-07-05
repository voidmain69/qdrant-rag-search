from app.models.search import MatchBranch
from app.services.code_index import CodeIndex

BOSCH_EAN = "3165140371940"  # valid EAN-13


def make_index() -> CodeIndex:
    idx = CodeIndex()
    idx.add_product("p1", {"article": "GSB-13-RE", "product_code": "060114E600", "ean13": BOSCH_EAN})
    idx.add_product("p2", {"article": "BOSCH-06", "product_code": None, "ean13": None})
    idx.add_product("p3", {"article": "HW-1234K", "product_code": "44556677", "ean13": None})
    return idx


class TestExact:
    def test_exact_with_separators(self):
        hits = make_index().match("gsb 13 re")
        assert hits and hits[0].point_id == "p1" and hits[0].branch == MatchBranch.EXACT

    def test_exact_ean(self):
        hits = make_index().match(BOSCH_EAN)
        assert hits and hits[0].branch == MatchBranch.EXACT and hits[0].field == "ean13"

    def test_cyrillic_homoglyph_exact(self):
        # НШ-1234К typed with Cyrillic Н, В, К maps onto Latin
        idx = CodeIndex()
        idx.add_product("p9", {"article": "HB-500T"})
        hits = idx.match("НВ-500Т")  # all-Cyrillic homoglyphs
        assert hits and hits[0].branch == MatchBranch.EXACT


class TestSkeleton:
    def test_ocr_confusion_found_via_skeleton(self):
        hits = make_index().match("BQSCH-O6")  # Q~O, O~0 visual confusion of BOSCH-06
        assert hits and hits[0].point_id == "p2"
        assert hits[0].branch == MatchBranch.EXACT_NORMALIZED


class TestEanCorrection:
    def test_single_wrong_digit(self):
        broken = BOSCH_EAN[:12] + "5"  # wrong check digit
        hits = make_index().match(broken)
        assert hits and hits[0].point_id == "p1"
        assert hits[0].branch == MatchBranch.EAN_CORRECTED

    def test_transposed_digits(self):
        s = list(BOSCH_EAN)
        i = next(i for i in range(12) if s[i] != s[i + 1])
        s[i], s[i + 1] = s[i + 1], s[i]
        hits = make_index().match("".join(s))
        assert hits and hits[0].point_id == "p1"
        assert hits[0].branch == MatchBranch.EAN_CORRECTED


class TestFuzzy:
    def test_typo_in_article(self):
        hits = make_index().match("GSB-13-RF")  # E -> F typo
        assert hits and hits[0].point_id == "p1"
        assert hits[0].branch == MatchBranch.FUZZY
        assert hits[0].score >= 0.8

    def test_missing_char(self):
        hits = make_index().match("HW-124K")  # dropped digit
        assert hits and hits[0].point_id == "p3"
        assert hits[0].branch == MatchBranch.FUZZY

    def test_garbage_rejected(self):
        assert make_index().match("XYZQWERTY99") == []
        assert make_index().match("пилосос") == []


class TestMutations:
    def test_remove_product(self):
        idx = make_index()
        idx.remove_product("p1")
        assert idx.match("GSB-13-RE") == [] or all(h.point_id != "p1" for h in idx.match("GSB-13-RE"))

    def test_reupsert_replaces_codes(self):
        idx = make_index()
        idx.add_product("p1", {"article": "NEW-9999"})
        assert all(h.branch != MatchBranch.EXACT for h in idx.match("GSB-13-RE"))
        hits = idx.match("NEW-9999")
        assert hits and hits[0].point_id == "p1" and hits[0].branch == MatchBranch.EXACT
