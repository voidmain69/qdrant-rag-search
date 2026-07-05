"""LLM product enrichment at ingest time (pre-embedding), via the configured backend.

The highest-leverage LLM step in the pipeline (see docs/pipeline_assessment.md): the
model sees the full product card once at ingest and extracts what search needs —
structured attributes for filters/coverage, uk/ru/en synonyms for BM25 recall, a
literal spec summary and use-cases for the dense vector. Paid once per product, never
per query; with the card in context the synonym task is far easier than blind
query-side translation (measured: gemma4:e4b mistranslates «безщітковий» from a bare
query but reads `Brushless` straight off a card).

Policy decisions:
* **Only products without structured attributes are enriched** — supplier attributes
  are ground truth, the LLM fills gaps, never competes with real data.
* **Product-provided fields always win** on merge (`build_payload`).
* **Graceful**: any LLM failure → the product is ingested unenriched; outcomes are
  visible in `ingest_enrichment_total{outcome}`.
* Throughput reality (~6–16 s/product on a local GPU; far faster on a cloud provider):
  fine for API upserts and background imports of gap-products; a full 100k-catalog
  enrichment is an offline batch job, not an inline step.

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

    def should_enrich(self, product: ProductIn) -> bool:
        """Supplier attributes are ground truth — the LLM only fills gaps."""
        return not product.attributes

    async def enrich(self, product: ProductIn) -> Enrichment | None:
        """Enrichment for one product, or None (skip / LLM failure) — ingestion
        must proceed unenriched in every failure mode."""
        if not self.should_enrich(product):
            ENRICHMENT_TOTAL.labels(outcome="skipped").inc()
            return None
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
        return list(await asyncio.gather(*(self.enrich(p) for p in products)))


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
