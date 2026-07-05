import httpx
import orjson

from app.core.config import Settings
from app.models.product import ProductIn
from app.services.coverage import fallback_requirements, requirements_coverage
from app.services.enrichment import Enrichment, ProductEnrichmentService, parse_enrichment
from app.services.ingest import build_payload as _build_payload
from app.services.llm import OllamaClient
from app.services.normalization import compose_dense_text, compose_sparse_text

DRILL = ProductIn(
    external_id="drl-1",
    name="Cordless Drill Bosch GSR 12V-30",
    brand="Bosch",
    description="Compact drill with brushless motor, 12V battery platform",
)

DRILL_ENRICHMENT = Enrichment(
    normalized_title="Bosch GSR 12V-30 cordless drill 12V brushless",
    synonyms=("шуруповерт", "дриль", "screwdriver", "безщітковий", "бесщеточный"),
    attributes={"motor": "brushless", "voltage": "12v", "cordless": True},
    use_cases=("домашній ремонт",),
    spec_summary="Cordless drill, 12V, brushless motor.",
)

LLM_ANSWER = {
    "normalized_title": "Bosch GSR 12V-30 cordless drill",
    "synonyms": ["шуруповерт", "безщітковий"],
    "attributes": {"Motor Type": "brushless", "voltage": 12, "cordless": True},
    "use_cases": ["домашній ремонт"],
    "spec_summary": "Cordless drill, 12V, brushless motor.",
}


def make_settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


def build_payload(product, dense_model, enrichment):
    """Test shim that supplies the content_hash argument."""
    return _build_payload(product, dense_model, enrichment, "test-hash")


class TestParseEnrichment:
    def test_valid_answer(self):
        e = parse_enrichment(orjson.dumps(LLM_ANSWER).decode())
        assert e is not None
        assert e.attributes == {"motor_type": "brushless", "voltage": 12, "cordless": True}
        assert "безщітковий" in e.synonyms
        assert e.spec_summary == "Cordless drill, 12V, brushless motor."

    def test_garbage(self):
        assert parse_enrichment("not json at all") is None
        assert parse_enrichment("[1, 2]") is None
        assert parse_enrichment("{}") is None

    def test_caps(self):
        raw = orjson.dumps(
            {"synonyms": [f"s{i}" for i in range(40)], "attributes": {f"k{i}": i for i in range(40)}}
        ).decode()
        e = parse_enrichment(raw)
        assert e is not None
        assert len(e.synonyms) <= 12
        assert len(e.attributes) <= 16


class TestMergePolicy:
    def test_product_attributes_win(self):
        product = ProductIn(external_id="x", name="y", attributes={"motor": "supplier-says-brushed"})
        enrichment = Enrichment(attributes={"motor": "brushless", "voltage": "12v"})
        payload = build_payload(product, "model-x", enrichment)
        assert payload["attributes"]["motor"] == "supplier-says-brushed"  # supplier wins
        assert payload["attributes"]["voltage"] == "12v"  # gap filled

    def test_enrichment_block_stored(self):
        payload = build_payload(DRILL, "model-x", DRILL_ENRICHMENT)
        assert payload["enrichment"]["synonyms"] == list(DRILL_ENRICHMENT.synonyms)
        assert payload["enrichment"]["spec_summary"] == DRILL_ENRICHMENT.spec_summary

    def test_no_enrichment_no_block(self):
        assert "enrichment" not in build_payload(DRILL, "model-x", None)


class TestTextComposition:
    def test_synonyms_only_in_sparse_text(self):
        dense = compose_dense_text(DRILL, DRILL_ENRICHMENT)
        sparse = compose_sparse_text(DRILL, DRILL_ENRICHMENT)
        assert "шуруповерт" not in dense  # synonyms must not dilute the dense vector
        assert "шуруповерт" in sparse
        assert "безщітковий" in sparse

    def test_spec_and_use_cases_in_dense_text(self):
        dense = compose_dense_text(DRILL, DRILL_ENRICHMENT)
        assert "brushless motor" in dense
        assert "домашній ремонт" in dense


class TestCoverageWithEnrichment:
    def test_ingest_alias_covers_query_without_query_llm(self):
        """The whole point: «безщітковий» is nowhere in the original card; the alias
        generated at ingest covers the plain token requirement — no query-time LLM."""
        payload = build_payload(DRILL, "model-x", DRILL_ENRICHMENT)
        requirements = fallback_requirements("безщітковий шуруповерт")
        assert requirements_coverage(requirements, payload) == (1.0, [])

    def test_unenriched_payload_still_misses(self):
        payload = build_payload(DRILL, "model-x", None)
        requirements = fallback_requirements("безщітковий шуруповерт")
        _ratio, missing = requirements_coverage(requirements, payload)
        assert missing == ["безщітковий", "шуруповерт"]


class TestEnrichService:
    def service(self, payload=None, status=200, calls=None, **settings_kw) -> ProductEnrichmentService:
        def handler(request: httpx.Request) -> httpx.Response:
            if calls is not None:
                calls.append(request)
            body = {"response": orjson.dumps(payload).decode()} if payload is not None else {}
            return httpx.Response(status, json=body)

        client = OllamaClient("http://ollama.test", transport=httpx.MockTransport(handler))
        return ProductEnrichmentService(client, make_settings(**settings_kw))

    async def test_attribute_rich_product_is_enriched_by_default(self):
        # default: every product gets synonyms (the cross-lingual recall lever); supplier
        # attributes still win on merge — see build_payload tests above
        calls: list[httpx.Request] = []
        svc = self.service(LLM_ANSWER, calls=calls)
        rich = ProductIn(external_id="x", name="y", attributes={"Сокет": "LGA 1200"})
        e = await svc.enrich(rich)
        assert e is not None and "безщітковий" in e.synonyms
        assert len(calls) == 1  # LLM was called for the attribute-rich product

    async def test_attribute_rich_skipped_when_flag_off(self):
        calls: list[httpx.Request] = []
        svc = self.service(LLM_ANSWER, calls=calls, ingest_enrich_with_attributes=False)
        rich = ProductIn(external_id="x", name="y", attributes={"Сокет": "LGA 1200"})
        assert await svc.enrich(rich) is None
        assert calls == []  # no LLM call for attribute-rich product

    async def test_gap_product_is_enriched(self):
        e = await self.service(LLM_ANSWER).enrich(DRILL)
        assert e is not None
        assert "безщітковий" in e.synonyms

    async def test_llm_error_returns_none(self):
        assert await self.service(status=500).enrich(DRILL) is None

    async def test_enrich_all_order(self):
        svc = self.service(LLM_ANSWER)
        rich = ProductIn(external_id="r", name="n", attributes={"a": "b"})
        results = await svc.enrich_all([rich, DRILL])
        assert results[0] is not None and results[1] is not None  # both enriched by default
