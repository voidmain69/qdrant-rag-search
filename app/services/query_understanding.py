"""LLM query understanding — dictionary-free synonymization via the configured backend.

Expands every significant token of a shopping query with lexical variants a product card
could literally contain: synonyms, Ukrainian/Russian/English translations, abbreviations
and format variants ("безщітковий" → "brushless", "1200" → "lga 1200"/"s1200"). No
hand-maintained dictionaries: the LLM generates the expansion per query and results are
LRU-cached.

The token list is computed by us and sent to the model — the LLM only *enriches* tokens
and can never drop a constraint (segmentation was measurably unreliable: gemma4:e4b
omitted «безщітковий» from "безщітковий шуруповерт makita" when asked to split the
query itself). Tokens absent from the answer degrade to single-variant requirements.

Failure policy: search must never depend on the LLM being up. Any error, timeout or
unparseable answer makes :meth:`QueryUnderstandingService.extract` return ``None`` and
the caller falls back to heuristic token requirements (`coverage.fallback_requirements`).

The backend (local Ollama or OpenAI-compatible) is injected as an :class:`LLMClient`;
this module is provider-agnostic. Chosen over SPLADE-style learned sparse expansion
because fastembed 0.8 ships no multilingual SPLADE checkpoint (see docs/search_semantics.md).
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict

import orjson
from prometheus_client import Counter

from app.core.config import Settings
from app.services.coverage import Requirement, significant_tokens
from app.services.llm import LLMClient, LLMError

logger = logging.getLogger(__name__)

UNDERSTANDING_TOTAL = Counter(
    "query_understanding_total", "LLM query understanding calls by outcome", ["outcome"]
)

MAX_VARIANTS = 8
MAX_VARIANT_LEN = 48
MAX_OUTPUT_TOKENS = 250
CACHE_SIZE = 1024

PROMPT_TEMPLATE = """\
You expand tokens of an e-commerce product search query (Ukrainian / Russian / English)
with lexical variants for exact matching against product cards.
For EVERY token give 2-6 variants that could literally appear in a product card:
synonyms, translations between Ukrainian, Russian and English, common abbreviations and formats.
Rules:
- variants must denote exactly the same thing as the token in this query's context; \
never broader or merely related terms (for "hdmi" the variant "port" is WRONG)
- each variant is 1-3 words, lowercase
- for numbers and codes keep common formats (e.g. "1200" -> "lga 1200", "s1200")
- prefer established technical terms: for "бездротовий" good variants are \
["wireless", "cordless", "беспроводной"], not transliterations
- respond with JSON only: an object mapping every token to its variant list

Query: {query}
Tokens: {tokens}"""


class QueryUnderstandingService:
    def __init__(self, client: LLMClient, settings: Settings):
        self._client = client
        self._model = settings.query_llm_model
        self._timeout = settings.query_llm_timeout_s
        self._cache: OrderedDict[str, list[Requirement]] = OrderedDict()
        # singleflight: coalesce concurrent identical cold queries onto one LLM call
        # instead of a thundering herd all missing the cache and all calling the model
        self._inflight: dict[str, asyncio.Task[list[Requirement] | None]] = {}

    async def warmup(self) -> None:
        await self._client.warmup(self._model)

    async def extract(self, query: str) -> list[Requirement] | None:
        """Requirements (one per significant query token, LLM-enriched variants), or
        None when the LLM path is unavailable — the caller must then degrade to
        heuristic token requirements."""
        tokens = significant_tokens(query)
        if not tokens:
            return []
        key = " ".join(tokens)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            UNDERSTANDING_TOTAL.labels(outcome="cache_hit").inc()
            return cached
        # ride an in-flight identical call if one is already running (no second LLM hit)
        running = self._inflight.get(key)
        if running is not None:
            return await running
        task = asyncio.ensure_future(self._compute(query, tokens, key))
        self._inflight[key] = task
        try:
            return await task
        finally:
            self._inflight.pop(key, None)

    async def _compute(self, query: str, tokens: list[str], key: str) -> list[Requirement] | None:
        prompt = PROMPT_TEMPLATE.replace("{query}", query).replace("{tokens}", orjson.dumps(tokens).decode())
        try:
            raw = await self._client.complete_json(
                prompt, model=self._model, max_tokens=MAX_OUTPUT_TOKENS, timeout=self._timeout
            )
        except LLMError:
            UNDERSTANDING_TOTAL.labels(outcome="error").inc()
            logger.warning("Query understanding failed for %r; using token fallback", query, exc_info=True)
            return None
        variants_map = parse_variants(raw)
        if variants_map is None:
            UNDERSTANDING_TOTAL.labels(outcome="unparseable").inc()
            logger.warning("Query understanding returned unusable JSON for %r", query)
            return None
        requirements = build_requirements(tokens, variants_map)
        UNDERSTANDING_TOTAL.labels(outcome="ok").inc()
        self._cache[key] = requirements
        if len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return requirements


def parse_variants(raw: str) -> dict[str, list[str]] | None:
    """Validate the LLM answer: a JSON object token → list of variant strings."""
    try:
        data = orjson.loads(raw)
    except orjson.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    out: dict[str, list[str]] = {}
    for token, raw_variants in data.items():
        if not isinstance(raw_variants, list):
            continue
        variants: list[str] = []
        for cand in raw_variants:
            text = str(cand).strip().lower()[:MAX_VARIANT_LEN]
            if text and text not in variants:
                variants.append(text)
            if len(variants) >= MAX_VARIANTS:
                break
        out[str(token).strip().lower()] = variants
    return out


def build_requirements(tokens: list[str], variants_map: dict[str, list[str]]) -> list[Requirement]:
    """One requirement per OUR token — the LLM can only add variants, never drop a
    constraint. The token itself is always the first variant."""
    requirements: list[Requirement] = []
    for token in tokens:
        variants = [token]
        for variant in variants_map.get(token, []):
            if variant not in variants:
                variants.append(variant)
        requirements.append(Requirement(name=token, variants=tuple(variants[:MAX_VARIANTS])))
    return requirements
