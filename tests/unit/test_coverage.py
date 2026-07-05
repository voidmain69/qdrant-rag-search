"""The motivating scenario: query "мат плата з hdmi на 1200" against a catalog where
no motherboard has both HDMI and socket 1200. Confident results must cover every
requested characteristic; near-misses are alternatives with missing_terms explaining why.
"""

from app.services.coverage import (
    Requirement,
    coverage,
    fallback_requirements,
    requirements_coverage,
    significant_tokens,
)

MB_WITH_HDMI = {
    "external_id": "mb-1",
    "name": "Материнська плата MSI B460M-A PRO",
    "brand": "MSI",
    "category": "Материнські плати",
    "attributes": {"Сокет": "LGA 1200", "Відеовиходи": "HDMI, DVI-D"},
}

MB_NO_HDMI = {
    "external_id": "mb-2",
    "name": "Материнська плата ASUS PRIME H410M-R",
    "brand": "ASUS",
    "category": "Материнські плати",
    "attributes": {"Сокет": "LGA 1200", "Відеовиходи": "D-Sub, DVI-D"},
}


class TestSignificantTokens:
    def test_stopwords_dropped(self):
        assert significant_tokens("мат плата з hdmi на 1200") == ["мат", "плата", "hdmi", "1200"]

    def test_punctuation_and_case(self):
        assert significant_tokens("Плата, HDMI!") == ["плата", "hdmi"]

    def test_only_stopwords(self):
        assert significant_tokens("з на для") == []


class TestCoverage:
    def test_all_terms_covered(self):
        tokens = significant_tokens("мат плата з hdmi на 1200")
        ratio, missing = coverage(tokens, MB_WITH_HDMI)
        assert ratio == 1.0
        assert missing == []

    def test_missing_hdmi_reported(self):
        tokens = significant_tokens("мат плата з hdmi на 1200")
        ratio, missing = coverage(tokens, MB_NO_HDMI)
        assert missing == ["hdmi"]
        assert ratio == 0.75

    def test_abbreviation_covers_full_word(self):
        # "мат" must cover "Материнська" (prefix match)
        assert coverage(["мат"], MB_WITH_HDMI) == (1.0, [])

    def test_inflection_tolerated(self):
        # "плати" (genitive) must cover "плата" via the dropped-last-char stem
        assert coverage(["плати"], MB_WITH_HDMI) == (1.0, [])

    def test_numeric_boundary_no_false_positive(self):
        # "1200" must NOT be satisfied by "12000"
        payload = {"name": "Блок живлення 12000 мАг"}
        _ratio, missing = coverage(["1200"], payload)
        assert missing == ["1200"]

    def test_numeric_matches_inside_compound(self):
        # "1200" must match "LGA1200" written without a space
        payload = {"name": "Плата", "attributes": {"Сокет": "LGA1200"}}
        assert coverage(["1200"], payload) == (1.0, [])

    def test_empty_tokens_vacuously_covered(self):
        assert coverage([], MB_WITH_HDMI) == (1.0, [])


class TestRequirementsCoverage:
    """Requirements carry LLM-generated variants — synonymization without dictionaries."""

    def test_multiword_variant(self):
        req = Requirement("материнська плата", ("материнська плата", "motherboard"))
        assert requirements_coverage([req], MB_WITH_HDMI) == (1.0, [])

    def test_synonym_variant_covers_english_attribute(self):
        # "безщітковий" is nowhere in the product — the "brushless" variant covers it
        drill = {
            "name": "Шуруповерт акумуляторний Makita DDF484Z",
            "attributes": {"Тип двигуна": "Brushless"},
        }
        req = Requirement("безщітковий", ("безщітковий", "brushless", "бесщеточный"))
        assert requirements_coverage([req], drill) == (1.0, [])

    def test_socket_format_variant(self):
        # LLM expands "на 1200" into formats; "lga 1200" matches the attribute value
        req = Requirement("сокет 1200", ("s1200", "lga 1200"))
        assert requirements_coverage([req], MB_WITH_HDMI) == (1.0, [])

    def test_missing_reports_requirement_name(self):
        req = Requirement("hdmi", ("hdmi", "hdmi порт"))
        ratio, missing = requirements_coverage([req], {"name": "Материнська плата"})
        assert ratio == 0.0
        assert missing == ["hdmi"]

    def test_fallback_requirements_mirror_tokens(self):
        reqs = fallback_requirements("мат плата з hdmi")
        assert [r.name for r in reqs] == ["мат", "плата", "hdmi"]
        assert all(r.variants == (r.name,) for r in reqs)
