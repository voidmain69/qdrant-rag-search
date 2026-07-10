"""LLM backend abstraction: local Ollama or OpenAI (/any OpenAI-compatible endpoint).

Both query understanding and ingest enrichment need one thing from an LLM: "given this
prompt, return a JSON string". :class:`LLMClient` is that single method; the concrete
subclasses adapt the two wire protocols (Ollama's ``/api/generate`` vs OpenAI's
``/chat/completions``). Callers stay provider-agnostic and select the backend via
``LLM_PROVIDER`` (see ``build_llm_client``).

Errors are normalized to :class:`LLMError` so callers can degrade gracefully without
importing httpx or knowing which backend is configured.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

import httpx

from app.core.config import LLMProvider, Settings

logger = logging.getLogger(__name__)

# keep an Ollama model resident between requests so only the first call pays the load cost
OLLAMA_KEEP_ALIVE = "30m"

# A transient blip must not permanently degrade a product (unenriched forever) or force a
# strict query onto the heuristic path — retry the genuinely transient failures with
# exponential backoff. Deliberately NOT retried: read/pool timeouts (the model is slow,
# not down — retrying only doubles latency; the caller degrades gracefully) and HTTP 4xx /
# 500 (a real error the same request won't fix). 429/502/503/504 and transport-level
# network errors are the retryable set.
LLM_RETRY_ATTEMPTS = 3
LLM_RETRY_BASE_DELAY_S = 0.5
_RETRYABLE_STATUS = frozenset({429, 502, 503, 504})


class LLMError(RuntimeError):
    """Any failure talking to the LLM backend (transport, HTTP status, bad body)."""


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    if isinstance(exc, httpx.ReadTimeout | httpx.PoolTimeout):
        return False  # slow model / saturated pool — retrying just doubles latency
    return isinstance(exc, httpx.TransportError)


async def _with_llm_retry(
    attempt: Callable[[], Awaitable[str]], *, what: str, max_attempts: int, base_delay: float
) -> str:
    """Run one LLM request attempt, retrying transient failures; map everything else to
    :class:`LLMError` so callers stay provider- and httpx-agnostic."""
    for i in range(max_attempts):
        try:
            return await attempt()
        except Exception as exc:
            if i < max_attempts - 1 and isinstance(exc, httpx.HTTPError) and _is_retryable(exc):
                delay = base_delay * 2**i
                logger.warning(
                    "LLM %s transient failure (attempt %d/%d), retrying in %.1fs: %s",
                    what,
                    i + 1,
                    max_attempts,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
                continue
            raise LLMError(f"{what} request failed: {exc}") from exc
    raise AssertionError("unreachable")


class LLMClient(ABC):
    # NOTE: `timeout` is a per-call budget forwarded straight to httpx (native support),
    # not an asyncio.timeout the caller manages — ASYNC109 is ignored for this file.
    @abstractmethod
    async def complete_json(self, prompt: str, *, model: str, max_tokens: int, timeout: float) -> str:
        """Return the model's raw response text (expected to be a JSON document).
        Raises :class:`LLMError` on any failure."""

    async def warmup(self, model: str) -> None:  # noqa: B027 — intentional optional no-op hook
        """Optional: preload the model so the first real call isn't cold. No-op by default."""

    @abstractmethod
    async def aclose(self) -> None: ...


class OllamaClient(LLMClient):
    def __init__(
        self,
        base_url: str,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        max_attempts: int = LLM_RETRY_ATTEMPTS,
        retry_base_delay: float = LLM_RETRY_BASE_DELAY_S,
    ):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(transport=transport)
        self._max_attempts = max_attempts
        self._retry_base_delay = retry_base_delay

    async def complete_json(self, prompt: str, *, model: str, max_tokens: int, timeout: float) -> str:
        async def _attempt() -> str:
            response = await self._client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "keep_alive": OLLAMA_KEEP_ALIVE,
                    "options": {"temperature": 0.0, "num_predict": max_tokens},
                },
                timeout=timeout,
            )
            response.raise_for_status()
            return str(response.json()["response"])

        return await _with_llm_retry(
            _attempt, what="Ollama", max_attempts=self._max_attempts, base_delay=self._retry_base_delay
        )

    async def warmup(self, model: str) -> None:
        try:
            await self._client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": "ok",
                    "stream": False,
                    "keep_alive": OLLAMA_KEEP_ALIVE,
                    "options": {"num_predict": 1},
                },
                timeout=300,
            )
            logger.info("Ollama model %s warmed up", model)
        except httpx.HTTPError:
            logger.warning("Ollama warmup failed (will retry lazily)", exc_info=True)

    async def aclose(self) -> None:
        await self._client.aclose()


class OpenAIClient(LLMClient):
    """OpenAI Chat Completions with JSON mode. Works against api.openai.com or any
    OpenAI-compatible server (vLLM, LiteLLM, Groq, local proxies) via OPENAI_BASE_URL."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        max_attempts: int = LLM_RETRY_ATTEMPTS,
        retry_base_delay: float = LLM_RETRY_BASE_DELAY_S,
    ):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(transport=transport, headers={"Authorization": f"Bearer {api_key}"})
        self._max_attempts = max_attempts
        self._retry_base_delay = retry_base_delay

    async def complete_json(self, prompt: str, *, model: str, max_tokens: int, timeout: float) -> str:
        async def _attempt() -> str:
            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.0,
                    "max_tokens": max_tokens,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            return str(response.json()["choices"][0]["message"]["content"])

        return await _with_llm_retry(
            _attempt, what="OpenAI", max_attempts=self._max_attempts, base_delay=self._retry_base_delay
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def build_llm_client(settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> LLMClient:
    if settings.llm_provider == LLMProvider.OPENAI:
        return OpenAIClient(settings.openai_base_url, settings.openai_api_key, transport)
    return OllamaClient(settings.ollama_url, transport)
