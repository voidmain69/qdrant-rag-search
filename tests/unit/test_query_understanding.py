import httpx
import orjson

from app.core.config import Settings
from app.services.llm import OllamaClient
from app.services.query_understanding import (
    QueryUnderstandingService,
    build_requirements,
    parse_variants,
)

LLM_ANSWER = {
    "мат": ["материнська", "материнская", "motherboard"],
    "плата": ["плата", "board", "mainboard"],
    "hdmi": ["hdmi"],
    "1200": ["lga 1200", "s1200"],
}


def make_settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


def make_service(payload=None, status=200, calls=None) -> QueryUnderstandingService:
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        body = {"response": orjson.dumps(payload).decode()} if payload is not None else {}
        return httpx.Response(status, json=body)

    client = OllamaClient("http://ollama.test", transport=httpx.MockTransport(handler))
    return QueryUnderstandingService(client, make_settings())


class TestParseVariants:
    def test_valid_answer(self):
        parsed = parse_variants(orjson.dumps(LLM_ANSWER).decode())
        assert parsed is not None
        assert parsed["1200"] == ["lga 1200", "s1200"]

    def test_garbage(self):
        assert parse_variants("motherboards are great") is None
        assert parse_variants('["a", "b"]') is None

    def test_non_list_values_skipped(self):
        parsed = parse_variants('{"hdmi": "hdmi port", "1200": ["lga 1200"]}')
        assert parsed == {"1200": ["lga 1200"]}

    def test_variants_normalized_and_capped(self):
        raw = orjson.dumps({"hdmi": [f"V{i}" for i in range(20)]}).decode()
        parsed = parse_variants(raw)
        assert parsed is not None
        assert len(parsed["hdmi"]) <= 8
        assert all(v == v.lower() for v in parsed["hdmi"])


class TestBuildRequirements:
    def test_token_is_always_first_variant(self):
        reqs = build_requirements(["безщітковий"], {"безщітковий": ["brushless", "бесщеточный"]})
        assert reqs[0].name == "безщітковий"
        assert reqs[0].variants == ("безщітковий", "brushless", "бесщеточный")

    def test_llm_cannot_drop_a_token(self):
        # the LLM answered only for one of two tokens — both must become requirements
        reqs = build_requirements(["безщітковий", "шуруповерт"], {"шуруповерт": ["screwdriver"]})
        assert [r.name for r in reqs] == ["безщітковий", "шуруповерт"]
        assert reqs[0].variants == ("безщітковий",)

    def test_llm_cannot_add_a_requirement(self):
        # extra keys in the answer must not create new constraints
        reqs = build_requirements(["hdmi"], {"hdmi": ["hdmi"], "wifi": ["wi-fi"]})
        assert [r.name for r in reqs] == ["hdmi"]


class TestExtract:
    async def test_ok_and_cached(self):
        calls: list[httpx.Request] = []
        svc = make_service(LLM_ANSWER, calls=calls)
        reqs = await svc.extract("мат плата з hdmi на 1200")
        assert reqs is not None
        assert [r.name for r in reqs] == ["мат", "плата", "hdmi", "1200"]
        assert "motherboard" in reqs[0].variants
        again = await svc.extract("  МАТ ПЛАТА з hdmi на 1200 ")  # same tokens → same cache key
        assert again == reqs
        assert len(calls) == 1

    async def test_stopword_only_query_needs_no_llm(self):
        calls: list[httpx.Request] = []
        svc = make_service(LLM_ANSWER, calls=calls)
        assert await svc.extract("з на для") == []
        assert calls == []

    async def test_http_error_returns_none(self):
        assert await make_service(status=500).extract("щось цікаве") is None

    async def test_unparseable_answer_returns_none(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": "oops"})

        client = OllamaClient("http://ollama.test", transport=httpx.MockTransport(handler))
        svc = QueryUnderstandingService(client, make_settings())
        assert await svc.extract("щось цікаве") is None
