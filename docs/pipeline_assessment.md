# Search pipeline assessment: what we adopt, what we already have, what we reject

Assessment of the proposed architecture (LLM enrichment → structured attributes →
hybrid retrieval → reranker → business rules) against this codebase, the uk/ru/en
catalog, and the actual host (RTX 3080 Ti, Ollama `gemma4:e4b`, fastembed/ONNX runtime).

Verdict summary:

| # | Proposed component | Verdict | Why |
|---|---|---|---|
| 0 | Pre-embedding LLM enrichment | **ADOPT** (top priority) | moves synonymy/structure work to ingest — paid once per product, not per query |
| 1 | Query parser → typed filters | **ADOPT after 0** | filters only work when payloads carry the same structured attributes enrichment produces |
| 2a | Hybrid Dense + BM25 + RRF | **ALREADY HAVE** | `query_points` with two prefetch branches, server-side RRF |
| 2b | + SPLADE third branch | **REJECT for now** | no multilingual SPLADE checkpoint in fastembed 0.8 (see `search_semantics.md`); BGE-M3 is the escalation path |
| 3 | Cross-encoder reranker | **ALREADY HAVE** | `bge-reranker-v2-m3`, optional group, lazy-loaded (`rerank=true`) |
| 4 | Post-rerank attribute filtering | **ALREADY HAVE (better)** | strict mode: violating hits are not discarded but moved to `alternatives` with `missing_terms` |
| — | Query rewriting / synonym expansion | **ALREADY HAVE** | strict-mode token expansion via Ollama (`query_understanding.py`) |
| — | Separate article/model index | **ALREADY HAVE (stronger)** | in-memory CodeIndex: exact → OCR-skeleton → EAN-checksum → RapidFuzz cascade beats BM25 on codes |
| — | Typo normalization | **PARTIAL** | codes: covered by skeleton/fuzzy tiers; free text: add to the enrichment/parse prompt later |
| — | Faceted search on extracted attributes | **ADOPT with 0+1** | needs enrichment attributes in payload + payload indexes |
| — | Learning-to-rank on clicks | **ROADMAP** | needs a click/purchase event stream first; log `search completed` events already carry query_kind/mode |

## 0. Pre-embedding enrichment — design decisions

The proposed JSON is rich; not all of it survives contact with a local 4B-class model.
Split by trust level:

**Adopt now (safe, high value):**
- `normalized_title` — factual, de-marketed title; goes into dense text.
- `synonyms` + `aliases` (несортовані «материнка», «mobo», name spellings) — go into the
  **sparse text only**: BM25 gets dictionary-free lexical recall («материнка» finds
  «Материнська плата») with zero query-time cost. Never into the dense text — e5 already
  handles semantic closeness; alias spam only dilutes the vector.
- `attributes` (typed key→value: socket=LGA1700, hdmi=true, form=mATX) — merged into
  `payload.attributes` (existing keys win — supplier data is ground truth, the LLM only
  fills gaps). This is what makes faceted filters and precise strict-mode coverage possible.
- `spec_summary` — literal technical sentence ("MicroATX motherboard, socket LGA1700,
  DDR4, 2×M.2, DisplayPort, HDMI"); appended to the dense text. This is the "embed the
  spec, not the marketing" idea and it is the main dense-quality lever.
- `use_cases` (ігровий ПК, NAS, офіс) — dense text; queries like «плата для домашнього
  NAS» hit nothing lexical today.

**Store but do not enforce (hallucination-prone):**
- `compatible_with` (CPU generations, PCIe, GPU models) — compatibility is precise
  domain knowledge; a 4B model guesses. Stored under `attributes` as informational,
  never used as a hard filter.

**Reject:**
- `negative_keywords` («що НЕ є товаром») — an LLM-invented negative can silently hide
  a valid product; the failure mode is invisible in production. Strict-mode coverage
  already expresses "does not satisfy X" positively via `missing_terms`.
- `brand_importance`, `search_phrases` — no consumer in the current ranking; skip until
  a use appears.
- A separate `embedding_text` field — we keep text composition in code
  (`normalization.py`): deterministic, testable, and the code/dense-text separation
  invariant stays enforced in one place. The LLM contributes *parts* (spec_summary,
  use_cases), not the final string.

**Throughput reality check** (measured ~6–16 s/call on gemma4:e4b under WSL2 GPU
paravirt — with wide variance and occasional multi-minute stalls / wedges on this
particular host):
- one-off catalog of 100k products ≈ days of local-GPU time → full-catalog enrichment
  is an offline batch job, not an inline step;
- therefore enrichment is **opt-in per deployment** (`INGEST_LLM_ENABLED`), applied
  inline for API upserts (1–1000 items — background imports already run async), always
  **graceful**: LLM failure/timeout → product is ingested unenriched, counted in
  `ingest_enrichment_total{outcome}`;
- **backend is pluggable** (`LLM_PROVIDER`, `app/services/llm.py`): the local model is
  the slow path. Pointing `LLM_PROVIDER=openai` (`gpt-4o-mini` or any OpenAI-compatible
  endpoint — vLLM, LiteLLM, Groq) turns enrichment from seconds into a fraction of a
  second and removes the local-GPU bottleneck for large catalogs. Same code path, same
  graceful degradation.

## 1. Query parser → filters: sequencing

«мат плата 1700 dp» → `{category: motherboard, socket: LGA1700, display_port: true}`
as a Qdrant `must` filter *before* vector search is the right end state, but it is
step 2, not step 1: filters can only match what payloads actually contain, i.e. the
attribute vocabulary produced by enrichment. Doing the parser first yields filters that
match nothing. Plan:

1. Enrichment ships and populates structured attributes (this change).
2. The existing strict-mode LLM call is extended to also emit
   `{attribute_filters: {...}}` using the same prompt (one call, not two); extracted
   filters are applied as soft boosts / strict-mode requirements first, and promoted to
   hard Qdrant `must` filters only for attributes with proven payload coverage.
3. Price/stock rules (`до 5000 грн`) are regex work, no LLM needed.

## GPU support (models actually using the 3080 Ti)

- **LLM (query understanding + enrichment)** — `LLM_PROVIDER=ollama` runs `gemma4:e4b`
  on the local GPU (verified: model resident in VRAM). WSL2 GPU paravirt makes it slow
  and occasionally unstable (multi-minute stalls observed); `LLM_PROVIDER=openai` offloads
  it entirely to a cloud/remote endpoint. The `LLMClient` abstraction (`app/services/llm.py`)
  makes this a one-env-var switch with no code change.
- **fastembed dense/sparse** — currently ONNX **CPU** inside the api container. CUDA
  path exists: `onnxruntime-gpu` + `providers=["CUDAExecutionProvider"]`
  (fastembed supports a `providers`/`cuda` argument) + `gpus` reservation in compose.
  Worth it for bulk ingest (10–30 prod/s → hundreds/s); query-time embedding is not the
  bottleneck (~30 ms). Requires a CUDA-enabled image (~+2 GB) — planned as an optional
  compose override, not the default image.
- **Reranker (`bge-reranker-v2-m3`)** — sentence-transformers/torch; on CPU it costs
  +50–200 ms per request, on GPU ~10×. Same compose-override story. Note: the host
  already runs TEI containers (`catalog-tei-*`) — text-embeddings-inference is the
  natural GPU serving path for embeddings+reranker if this service outgrows in-process
  models; that is an infra decision, the model-profile abstraction keeps the app code
  agnostic.

## What stays deliberately unchanged

- **CodeIndex over BM25 for артикули** — the proposal suggests BM25/inverted index for
  codes; our dedicated cascade (exact → homoglyph → OCR skeleton → EAN checksum →
  RapidFuzz) is strictly stronger and already short-circuits vector search.
- **RRF fusion** — stays server-side in Qdrant, no hand-tuned weights.
- **No LLM on the per-query hot path** in relaxed mode; strict mode pays one cached
  LLM call — matches the proposal's "LLM тільки на імпорті, швидкі компоненти на запиті".
