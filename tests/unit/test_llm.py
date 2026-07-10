import httpx
import orjson
import pytest

from app.core.config import Settings
from app.services.llm import (
    LLM_RETRY_ATTEMPTS,
    LLMError,
    OllamaClient,
    OpenAIClient,
    build_llm_client,
)


def make_settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


def _counting_transport(responses):
    """MockTransport that returns/raises the next item per call; the last item repeats.
    A callable item is invoked to raise (e.g. httpx.ConnectError)."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        item = responses[min(len(calls), len(responses) - 1)]
        calls.append(request)
        if callable(item):
            raise item()
        return item

    return httpx.MockTransport(handler), calls


class TestOllamaClient:
    async def test_returns_response_field(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["json"] = orjson.loads(request.content)
            return httpx.Response(200, json={"response": '{"ok": true}'})

        client = OllamaClient("http://ollama.test", transport=httpx.MockTransport(handler))
        out = await client.complete_json("hi", model="gemma4:e4b", max_tokens=100, timeout=5)
        assert out == '{"ok": true}'
        assert captured["url"] == "http://ollama.test/api/generate"
        assert captured["json"]["model"] == "gemma4:e4b"
        assert captured["json"]["options"]["num_predict"] == 100
        await client.aclose()

    async def test_http_error_becomes_llm_error(self):
        client = OllamaClient("http://x", transport=httpx.MockTransport(lambda r: httpx.Response(500)))
        with pytest.raises(LLMError):
            await client.complete_json("hi", model="m", max_tokens=10, timeout=5)
        await client.aclose()


class TestOpenAIClient:
    async def test_chat_completions_shape_and_auth(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["auth"] = request.headers.get("authorization")
            captured["json"] = orjson.loads(request.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]})

        client = OpenAIClient("https://api.openai.com/v1", "sk-test", transport=httpx.MockTransport(handler))
        out = await client.complete_json("hi", model="gpt-4o-mini", max_tokens=100, timeout=5)
        assert out == '{"ok": true}'
        assert captured["url"] == "https://api.openai.com/v1/chat/completions"
        assert captured["auth"] == "Bearer sk-test"
        assert captured["json"]["response_format"] == {"type": "json_object"}
        assert captured["json"]["model"] == "gpt-4o-mini"
        await client.aclose()

    async def test_http_error_becomes_llm_error(self):
        client = OpenAIClient("https://x", "k", transport=httpx.MockTransport(lambda r: httpx.Response(401)))
        with pytest.raises(LLMError):
            await client.complete_json("hi", model="m", max_tokens=10, timeout=5)
        await client.aclose()

    async def test_malformed_body_becomes_llm_error(self):
        client = OpenAIClient(
            "https://x", "k", transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))
        )
        with pytest.raises(LLMError):
            await client.complete_json("hi", model="m", max_tokens=10, timeout=5)
        await client.aclose()


class TestRetry:
    async def test_retries_transient_status_then_succeeds(self):
        transport, calls = _counting_transport(
            [httpx.Response(503), httpx.Response(200, json={"response": "{}"})]
        )
        client = OllamaClient("http://x", transport=transport, retry_base_delay=0.0)
        out = await client.complete_json("hi", model="m", max_tokens=10, timeout=5)
        assert out == "{}"
        assert len(calls) == 2  # first 503 retried, second succeeded
        await client.aclose()

    async def test_retries_transport_error_then_succeeds(self):
        transport, calls = _counting_transport(
            [lambda: httpx.ConnectError("boom"), httpx.Response(200, json={"response": "{}"})]
        )
        client = OllamaClient("http://x", transport=transport, retry_base_delay=0.0)
        out = await client.complete_json("hi", model="m", max_tokens=10, timeout=5)
        assert out == "{}"
        assert len(calls) == 2
        await client.aclose()

    async def test_persistent_transient_exhausts_attempts(self):
        transport, calls = _counting_transport([httpx.Response(503)])
        client = OllamaClient("http://x", transport=transport, retry_base_delay=0.0)
        with pytest.raises(LLMError):
            await client.complete_json("hi", model="m", max_tokens=10, timeout=5)
        assert len(calls) == LLM_RETRY_ATTEMPTS  # tried, backed off, gave up
        await client.aclose()

    async def test_non_transient_status_not_retried(self):
        # 500 is a hard error (a real fault the same request won't fix), not retried
        transport, calls = _counting_transport([httpx.Response(500)])
        client = OllamaClient("http://x", transport=transport, retry_base_delay=0.0)
        with pytest.raises(LLMError):
            await client.complete_json("hi", model="m", max_tokens=10, timeout=5)
        assert len(calls) == 1
        await client.aclose()

    async def test_read_timeout_not_retried(self):
        # a slow model (read timeout) is not down — retrying just doubles latency
        transport, calls = _counting_transport([lambda: httpx.ReadTimeout("slow")])
        client = OllamaClient("http://x", transport=transport, retry_base_delay=0.0)
        with pytest.raises(LLMError):
            await client.complete_json("hi", model="m", max_tokens=10, timeout=5)
        assert len(calls) == 1
        await client.aclose()


class TestBuildLLMClient:
    def test_default_is_ollama(self):
        assert isinstance(build_llm_client(make_settings()), OllamaClient)

    def test_openai_selected(self):
        settings = make_settings(llm_provider="openai", openai_api_key="sk-x")
        assert isinstance(build_llm_client(settings), OpenAIClient)


class TestOpenAIGuard:
    def test_openai_without_key_and_feature_rejected(self):
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            make_settings(llm_provider="openai", query_llm_enabled=True, openai_api_key="")

    def test_openai_without_key_but_no_feature_ok(self):
        # provider set but no LLM feature enabled → no key required
        make_settings(llm_provider="openai", openai_api_key="")

    def test_openai_with_key_ok(self):
        make_settings(llm_provider="openai", ingest_llm_enabled=True, openai_api_key="sk-x")
