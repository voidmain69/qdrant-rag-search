"""LLM product enrichment at ingest time (pre-embedding), via the configured backend.

The highest-leverage LLM step in the pipeline (see docs/pipeline_assessment.md): the
model sees the full product card once at ingest and extracts what search needs —
structured attributes for filters/coverage, uk/ru/en synonyms for BM25 recall, a
literal spec summary and use-cases for the dense vector. Paid once per product, never
per query; with the card in context the synonym task is far easier than blind
query-side translation (measured: gemma4:e4b mistranslates «безщітковий» from a bare
query but reads `Brushless` straight off a card).

Policy decisions:
* **Every product is enriched** by default — the biggest win is the uk/ru/en synonyms,
  which help cross-lingual recall regardless of whether the product has attributes (a
  3-way benchmark showed e5+BM25+enrichment beating raw BGE-M3 precisely because of these
  synonyms). Set `INGEST_ENRICH_WITH_ATTRIBUTES=false` to enrich only attribute-less
  products (one fewer LLM call per attribute-rich product).
* **Supplier fields always win** on merge (`build_payload`) — the LLM only fills gaps,
  never overwrites real attributes.
* **Graceful**: any LLM failure → the product is ingested unenriched; outcomes are
  visible in `ingest_enrichment_total{outcome}`.
* Throughput reality (~6–16 s/product on a local GPU; far faster on a cloud provider):
  a one-time cost — `content_hash` skips re-enriching unchanged products on re-push.

The backend (local Ollama or OpenAI-compatible) is injected as an :class:`LLMClient`.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import orjson
from prometheus_client import Counter

from app.core.config import Settings
from app.models.product import AttrValue, ProductIn
from app.services.llm import LLMClient, LLMError

logger = logging.getLogger(__name__)

ENRICHMENT_TOTAL = Counter(
    "ingest_enrichment_total", "LLM product enrichment attempts by outcome", ["outcome"]
)

MAX_SYNONYMS = 12
MAX_USE_CASES = 6
MAX_ATTRIBUTES = 16
MAX_VALUE_LEN = 80
MAX_TEXT_LEN = 400
MAX_OUTPUT_TOKENS = 600

PROMPT_TEMPLATE = """\
You are a product-data analyst for an e-commerce search engine (Ukrainian / Russian / English catalog).
Extract search-relevant structure from the product card below. Use ONLY facts present in the card;
never invent characteristics.
Respond with JSON only:
{
 "normalized_title": "<short factual title: brand, model, key specs; no marketing>",
 "synonyms": ["<uk/ru/en names a shopper could type for this product type and its key traits>"],
 "attributes": {"<latin_snake_case_key>": <short string, number or boolean>},
 "use_cases": ["<short scenario, e.g. 'домашній сервер'>"],
 "spec_summary": "<one literal sentence listing the technical specs>"
}
Rules:
- synonyms: include Ukrainian, Russian and English forms («материнська плата», «материнка», "motherboard");
  translate technical terms found in the card (e.g. "Brushless" -> «безщітковий», «бесщеточный»)
- attributes: only objective filterable facts (socket, form factor, voltage, ports, wireless, ...)
- everything lowercase except proper names; no duplicates; empty list/object when nothing applies

Product card:
{card}"""


@dataclass(frozen=True)
class Enrichment:
    normalized_title: str | None = None
    synonyms: tuple[str, ...] = ()
    attributes: dict[str, AttrValue] = field(default_factory=dict)
    use_cases: tuple[str, ...] = ()
    spec_summary: str | None = None


def _product_card(p: ProductIn) -> str:
    lines = [f"Name: {p.name}"]
    if p.brand:
        lines.append(f"Brand: {p.brand}")
    if p.category:
        lines.append(f"Category: {p.category}")
    if p.article:
        lines.append(f"Article: {p.article}")
    if p.description:
        lines.append(f"Description: {p.description[:1500]}")
    return "\n".join(lines)


class ProductEnrichmentService:
    def __init__(self, client: LLMClient, settings: Settings):
        self._client = client
        self._model = settings.ingest_llm_model
        self._timeout = settings.ingest_llm_timeout_s
        self._semaphore = asyncio.Semaphore(settings.ingest_llm_concurrency)
        self._enrich_with_attributes = settings.ingest_enrich_with_attributes

    def should_enrich(self, product: ProductIn) -> bool:
        """Enrich every product for its synonyms/spec/use-cases; supplier attributes
        still win on merge. Unless INGEST_ENRICH_WITH_ATTRIBUTES=false, then only
        attribute-less products are enriched."""
        return self._enrich_with_attributes or not product.attributes

    async def enrich(self, product: ProductIn) -> Enrichment | None:
        """Enrichment for one product, or None (skip / LLM failure) — ingestion
        must proceed unenriched in every failure mode."""
        if not self.should_enrich(product):
            ENRICHMENT_TOTAL.labels(outcome="skipped").inc()
            return None
        return await self._enrich_card(product)

    async def _enrich_card(self, product: ProductIn) -> Enrichment | None:
        """One LLM call for one product card (caller has already decided it should be
        enriched). Every failure mode returns None so ingestion proceeds unenriched."""
        prompt = PROMPT_TEMPLATE.replace("{card}", _product_card(product))
        try:
            async with self._semaphore:
                raw = await self._client.complete_json(
                    prompt, model=self._model, max_tokens=MAX_OUTPUT_TOKENS, timeout=self._timeout
                )
        except LLMError:
            ENRICHMENT_TOTAL.labels(outcome="error").inc()
            logger.warning(
                "Enrichment failed for %r; ingesting unenriched", product.external_id, exc_info=True
            )
            return None
        enrichment = parse_enrichment(raw)
        if enrichment is None:
            ENRICHMENT_TOTAL.labels(outcome="unparseable").inc()
            logger.warning("Enrichment returned unusable JSON for %r", product.external_id)
            return None
        ENRICHMENT_TOTAL.labels(outcome="ok").inc()
        return enrichment

    async def enrich_all(self, products: list[ProductIn]) -> list[Enrichment | None]:
        """Enrich a batch, deduplicating identical product cards: enrichment depends only
        on the card text (name/brand/category/article/description), so N products with the
        same card cost one LLM call, not N. Common on re-push / near-duplicate variants."""
        cards: list[str | None] = []  # per input product: its card key, or None if skipped
        to_run: dict[str, ProductIn] = {}  # unique card → representative product
        for product in products:
            if not self.should_enrich(product):
                ENRICHMENT_TOTAL.labels(outcome="skipped").inc()
                cards.append(None)
                continue
            card = _product_card(product)
            to_run.setdefault(card, product)
            cards.append(card)
        if len(to_run) < sum(1 for c in cards if c is not None):
            logger.info(
                "enrichment: %d unique cards across %d products (deduped)", len(to_run), len(products)
            )
        order = list(to_run)
        results = await asyncio.gather(*(self._enrich_card(to_run[card]) for card in order))
        by_card = dict(zip(order, results, strict=True))
        return [None if card is None else by_card[card] for card in cards]


def _clean_str_list(raw: object, limit: int, max_len: int = MAX_VALUE_LEN) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    out: list[str] = []
    for item in raw:
        text = str(item).strip()[:max_len]
        if text and text.lower() not in {o.lower() for o in out}:
            out.append(text)
        if len(out) >= limit:
            break
    return tuple(out)


def parse_enrichment(raw: str) -> Enrichment | None:
    """Validate and cap the LLM answer; None when nothing usable came back."""
    try:
        data = orjson.loads(raw)
    except orjson.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    attributes: dict[str, AttrValue] = {}
    raw_attrs = data.get("attributes")
    if isinstance(raw_attrs, dict):
        for key, value in raw_attrs.items():
            if len(attributes) >= MAX_ATTRIBUTES:
                break
            key_str = str(key).strip().lower().replace(" ", "_")[:MAX_VALUE_LEN]
            if not key_str:
                continue
            if isinstance(value, bool | int | float):
                attributes[key_str] = value
            else:
                value_str = str(value).strip()[:MAX_VALUE_LEN]
                if value_str:
                    attributes[key_str] = value_str

    def _clean_text(key: str) -> str | None:
        value = str(data.get(key) or "").strip()[:MAX_TEXT_LEN]
        return value or None

    enrichment = Enrichment(
        normalized_title=_clean_text("normalized_title"),
        synonyms=_clean_str_list(data.get("synonyms"), MAX_SYNONYMS),
        attributes=attributes,
        use_cases=_clean_str_list(data.get("use_cases"), MAX_USE_CASES),
        spec_summary=_clean_text("spec_summary"),
    )
    if (
        not enrichment.synonyms
        and not enrichment.attributes
        and not enrichment.use_cases
        and enrichment.spec_summary is None
        and enrichment.normalized_title is None
    ):
        return None
    return enrichment
