# Qdrant Product Search

A production-grade **product search service** built on [Qdrant](https://qdrant.tech/). It combines semantic vector search, lexical BM25, and an exact/fuzzy code-matching engine behind a single REST API, so one endpoint correctly answers every kind of query a product catalog receives:

| Query example | What the user meant | How it is answered |
|---|---|---|
| `бездротовий пилосос для дому` | a natural-language phrase | dense + sparse hybrid search with server-side RRF fusion |
| `акумулятор 18V 5Ah Li-Ion` | product characteristics | hybrid search — measurement tokens (`18V`, `5Ah`) are recognized as units, **not** product codes |
| `GSB-13-RE` | an exact article / SKU | O(1) exact lookup, ~3 ms |
| `НВ-500Т` (Cyrillic letters) | a SKU typed in a Ukrainian keyboard layout | Cyrillic→Latin homoglyph normalization (`Н→H`, `В→B`, `Т→T`) |
| `BQSCH-O6` | an OCR-mangled code | lossy OCR-skeleton matching (`O↔0`, `I↔1`, `S↔5`, `B↔8`, …) |
| `4006381333932` | an EAN-13 with one wrong digit | checksum-aware error correction (single substitution / adjacent transposition) |
| `GSB-13-RF` | a typo in an article | RapidFuzz cascade (OSA + Jaro-Winkler) over an in-memory code corpus |
| `дриль GSB13RE з кейсом` | code + free text | both branches run; exact code hits are pinned first |

Every hit carries a `match` explanation (`branch`, `matched_field`, `code_score`), so clients can always see *why* a product ranked where it did.

The service exposes two API surfaces:

1. **Ingestion API** — push products (single / batch / file import) for normalization, vectorization, and upsert into Qdrant.
2. **Search API** — one intelligent search endpoint with filters, pagination, and optional cross-encoder reranking.

There is deliberately **no LLM answer generation**: this is a search engine, not a chatbot. Catalog data is multilingual (Ukrainian / Russian / English).

---

## Table of contents

- [Architecture](#architecture)
- [Key flows](#key-flows)
- [API contract](#api-contract)
- [Request / response examples](#request--response-examples)
- [Deployment](#deployment)
- [Configuration](#configuration)
- [Development](#development)
- [Design decisions & limitations](#design-decisions--limitations)

---

## Architecture

```
                                ┌──────────────────────────────────────────────┐
                                │                 FastAPI (api)                │
                                │                                              │
 POST /api/v1/products ───────► │  IngestService                               │
 POST /api/v1/products:batch    │   normalize ─► embed (dense+sparse) ─► upsert│───► Qdrant
 POST /api/v1/imports (files)   │   └─► CodeIndex.add (incremental)            │     collection "products"
                                │                                              │      ├ named vector  "dense"       (1024, cosine)
 POST /api/v1/search ─────────► │  SearchService                               │      ├ sparse vector "sparse_text" (BM25 + IDF)
                                │   classify(query)                            │      └ payload indexes (keyword/float/bool/text)
                                │   ├─ CODE_ONLY ─► CodeIndex (in-memory)      │
                                │   │    exact → skeleton → EAN-fix → fuzzy    │     collection "service_meta"
                                │   ├─ TEXT ──────► query_points(              │      └ embedding model / dim / schema version
                                │   │    prefetch=[dense, sparse], RRF fusion) │
                                │   │    └─ optional cross-encoder rerank      │
                                │   └─ MIXED ─────► both, exact hits pinned    │
                                └──────────────────────────────────────────────┘
```

### Components

| Component | File | Responsibility |
|---|---|---|
| **Query router** | `app/services/query_router.py` | Classifies queries (`code_only` / `text` / `mixed`), orchestrates branches, merges results |
| **Code index** | `app/services/code_index.py` | In-memory exact/fuzzy index over all article / product-code / EAN values |
| **Normalization** | `app/services/normalization.py` | Code canonicalization, homoglyph mapping, OCR skeleton, embedding-text composition |
| **EAN engine** | `app/services/ean.py` | EAN-13 checksum validation and error-candidate generation |
| **Embeddings** | `app/services/embedding.py` | Local dense + sparse embeddings via FastEmbed (ONNX, CPU), model profiles |
| **Qdrant service** | `app/services/qdrant.py` | Collection schema, payload indexes, hybrid RRF query, filter builder |
| **Ingest** | `app/services/ingest.py` | normalize → embed → upsert pipeline, deterministic point IDs |
| **Importer** | `app/services/importer.py` | CSV / JSON / XLSX parsing, column mapping, background jobs |
| **Reranker** | `app/services/reranker.py` | Optional lazy-loaded cross-encoder (`BAAI/bge-reranker-v2-m3`) |

### Models

- **Dense**: `intfloat/multilingual-e5-large` (FastEmbed/ONNX, CPU, 1024-dim, cosine). Strong uk/ru/en semantics; `query:` / `passage:` prefixes are applied by a model-profile abstraction, so the model can be swapped via config.
- **Sparse**: `Qdrant/bm25` with the Russian Snowball stemmer (no Ukrainian stemmer exists in FastEmbed; Ukrainian and English tokens still match verbatim). Requires the `IDF` modifier on the Qdrant sparse vector — configured automatically.
- **Fusion**: native Qdrant Query API — `query_points` with two `prefetch` branches fused by **Reciprocal Rank Fusion** on the server. No hand-tuned score weights.
- **Reranker** (optional): `BAAI/bge-reranker-v2-m3` cross-encoder via sentence-transformers, lazy-loaded on the first `rerank=true` request.

### Text composition (what gets embedded)

- **Dense text**: `name + brand + category + attributes + description`. Codes/SKU/EAN are **excluded** so alphanumeric noise never pollutes the semantic vector.
- **Sparse text**: dense text **plus** all code fields in both raw and normalized form — the hybrid branch gets lexical exact-match power on codes for free.

### Code matching cascade

For each code-like query token (first non-empty tier wins):

| Tier | Mechanism | Score | Branch |
|---|---|---|---|
| 1 | exact match on normalized form (NFKC, uppercase, separators stripped, Cyrillic homoglyphs → Latin) | 1.00 | `exact` |
| 2 | exact match on lossy OCR skeleton (`O→0, Q→0, D→0, I→1, L→1, Z→2, S→5, G→6, B→8, З→3`) | 0.97 | `exact_normalized` |
| 3 | checksum-valid EAN-13 candidates one digit-substitution or adjacent transposition away; also UPC-A→EAN-13 promotion and GTIN-14 handling | 0.95 | `ean_corrected` |
| 4 | RapidFuzz `process.extract` over the whole corpus: OSA similarity cutoff 0.72, re-scored `0.6·OSA + 0.4·JaroWinkler − length penalty`, accepted at ≥ 0.80 | 0.80–0.95 | `fuzzy` |

---

## Key flows

### 1. Ingestion

```
ProductIn (Pydantic validation: whitespace collapse, EAN digits-only)
  → point_id = uuid5(namespace, external_id)          # deterministic → idempotent upserts
  → compose dense text (no codes) + sparse text (with codes)
  → FastEmbed batch embedding (batch 64, off the event loop)
  → Qdrant upsert (chunks of 256, wait=true)
  → CodeIndex.add_product (incremental, thread-safe)
```

Re-ingesting the same `external_id` **replaces** the product (same point ID). `PUT` and `POST` converge on the same upsert path.

### 2. Search — `text` query

```
"бездротовий пилосос для дому"
  → classify: no code-like tokens → TEXT
  → embed query (dense + sparse)
  → Qdrant query_points:
      prefetch: [dense (limit 50, filtered), sparse_text (limit 50, filtered)]
      query:    FusionQuery(RRF)
  → optional: cross-encoder rerank of top-50 (rerank=true)
  → SearchResponse (branch = "hybrid")
```

### 3. Search — `code_only` query

```
"4006381333932"                       # EAN with a wrong check digit
  → classify: single code-like token → CODE_ONLY
  → CodeIndex.match: exact ✗ → skeleton ✗ → EAN candidates: {4006381333931✓}
  → strong hit (score ≥ 0.95) → short-circuit, NO vector search at all (~3 ms)
  → fetch payloads by point id, apply filters in-app
  → SearchResponse (branch = "ean_corrected")
```

If only weak/fuzzy hits are found, the router **falls through** to hybrid search and pins the fuzzy code hits above the semantic results.

### 4. Search — `mixed` query

```
"дриль GSB13RE з кейсом"
  → classify: code tokens ["GSB13RE"] + text → MIXED
  → branch A: CodeIndex.match("GSB13RE") → exact hit
  → branch B: hybrid search over the full query (codes stay in the sparse text)
  → merge: exact/skeleton/EAN hits pinned first, then fuzzy + hybrid interleaved (dedup by product)
```

### 5. File import

```
POST /api/v1/imports (multipart: file [+ column_mapping JSON])
  → 202 Accepted { job_id }
  → background: parse CSV/JSON/XLSX → map columns → validate rows → ingest in chunks of 200
  → GET /api/v1/imports/{job_id} → { status, total, processed, failed, errors[≤100] }
```

Column mapping: explicit `column_mapping` > built-in aliases (`sku`/`артикул`→`article`, `ean`/`barcode`/`штрихкод`→`ean13`, `назва`/`название`→`name`, `ціна`/`цена`→`price`, …) > exact field name. Unknown columns land in `attributes`.

### 6. Schema versioning / reindex

A one-point companion collection `service_meta` records `{dense_model, dense_dim, sparse_model, schema_version}`. On startup the service compares it with the current config and **refuses to start** on mismatch — you can never silently mix vectors from different models. To reindex: drop the `products` and `service_meta` collections, restart, re-ingest.

---

## API contract

Base URL: `http://<host>:8000`. OpenAPI/Swagger UI: **`/docs`**.

**Authentication**: header `X-API-Key: <key>` on everything under `/api/v1`. Keys come from the `API_KEYS` env var (comma-separated, constant-time compared). An empty `API_KEYS` disables auth — dev only. `/health` and `/ready` are always open.

| Method & path | Purpose | Success |
|---|---|---|
| `POST /api/v1/products` | Upsert one product | `200` → batch result |
| `POST /api/v1/products:batch` | Upsert up to 1000 products (synchronous) | `200` → batch result |
| `POST /api/v1/products:batch-async` | Enqueue a batch upsert as a background job | `202` → job |
| `PUT /api/v1/products/{external_id}` | Full replace (body `external_id` must match path) | `200` |
| `PATCH /api/v1/products/{external_id}/price` | Update only price / availability — no re-embedding | `200`; `404` if absent |
| `POST /api/v1/products:prices` | Bulk price / availability update (≤1000, partial success) | `200` → batch result |
| `POST /api/v1/products:archive` | Bulk archive (hide, retain) or restore (`archived:false`) | `200` → batch result |
| `POST /api/v1/products:delete` | Bulk hard delete (partial success) | `200` → batch result |
| `POST /api/v1/products:reconcile` | Snapshot sync: archive everything not in `external_ids` (dry-run + cap) | `200`; `400` past cap |
| `POST /api/v1/products:diff` | Drift report vs a snapshot (no mutation) | `200` |
| `GET /api/v1/products/stats` | `{total, active, archived}` | `200` |
| `DELETE /api/v1/products/{external_id}` | Delete one | `204`; `404` if absent |
| `POST /api/v1/imports` | Start file import (multipart `file`, optional form field `column_mapping`) | `202` → job |
| `GET /api/v1/imports/{job_id}` | Import progress | `200` → job |
| `POST /api/v1/search` | Search | `200` → results |
| `GET /health` | Liveness | always `200` |
| `GET /ready` | Readiness (Qdrant reachable, models loaded, CodeIndex built) | `200` / `503` |
| `GET /metrics` | Prometheus metrics (if `METRICS_ENABLED=true`) | `200` |

Common errors: `401` invalid/missing API key, `422` validation error (Pydantic detail body), `400` rerank requested but disabled, `413` import file > 100 MB.

**Price / availability updates.** A full upsert re-runs the whole heavy pipeline (LLM enrichment + dense/sparse embedding), which is wasted work when only `price`, `in_stock` or `currency` change — those fields are not embedded. Use `PATCH /products/{id}/price` (single) or `POST /products:prices` (bulk) instead: they patch the Qdrant payload directly (`set_payload`, vectors untouched, CodeIndex untouched), completing in milliseconds. Body is a `PriceUpdate` — `external_id` plus at least one of `price` / `in_stock` / `currency`; only the provided fields change, and `updated_at` is bumped. The bulk endpoint is partial-success: unknown `external_id`s come back as failed items (`"Product not found"`) rather than failing the batch.

### Catalog sync (keeping the index in step with your source of truth)

An upstream system (PIM / ERP / 1C) owns the catalog; this service is a downstream search/RAG index that must be kept in sync. Three lifecycle levers, deliberately orthogonal:

- **`in_stock`** — buyable right now? Toggle via `:prices`. The product stays searchable and shows as unavailable.
- **`status = archived`** — temporarily out of the catalog: hidden from search by default, **retained** (vectors kept), reversible. Set via `POST /products:archive` (`archived:true`) / restore with `archived:false`, or fetch anyway with `include_archived:true` on `/search`.
- **delete** — gone for good, vectors removed. `DELETE /products/{id}` or bulk `POST /products:delete`.

**Re-push everything, cheaply.** `:batch` stores a `content_hash` of each product's embedding-affecting fields (name/description/brand/category/codes/attributes). Re-uploading a product whose searchable content didn't change **skips embedding and enrichment** entirely and just refreshes the payload — so a nightly full re-push only pays the LLM/ONNX cost for what actually changed. Force a re-embed (e.g. after changing the enrichment prompt) with `REEMBED_UNCHANGED=true`.

**Removing what the source forgot to delete (orphans).** If a delete event is lost, the index keeps a product the source no longer has. `POST /products:reconcile` fixes drift from a snapshot: send the full list of currently-valid `external_id`s, and everything **not** in it is **archived** (never deleted — a truncated snapshot is recoverable). Safety rails: `dry_run` (default `true`) reports what *would* be archived without touching anything, and `max_archived` refuses the run (400) if it would archive more than N products — so a broken source feed can't wipe the catalog. `POST /products:diff` gives the same drift report (`missing_in_index` / `extra_in_index`) with no mutation, and `GET /products/stats` returns `{total, active, archived}` for a cheap divergence check.

```bash
# nightly full sync from the source of truth:
POST /api/v1/products:batch      { "items": [ ...whole catalog... ] }   # adds/updates; unchanged skip embedding
POST /api/v1/products:reconcile  { "external_ids": [...all valid ids...], "dry_run": false, "max_archived": 5000 }
```

### Product schema (`ProductIn`)

```jsonc
{
  "external_id": "tool-001",         // required, ≤128 chars — stable ID in your system
  "name": "Дриль ударний Bosch GSB 13 RE",  // required, ≤512 chars
  "description": "…",                // optional, ≤10 000 chars
  "brand": "Bosch",                  // optional
  "category": "Електроінструмент",   // optional
  "article": "GSB-13-RE",            // optional — SKU / артикул
  "product_code": "060114E600",      // optional — internal product code
  "ean13": "4006381333931",          // optional — non-digits are stripped
  "attributes": {"Потужність": "600 Вт", "Патрон": "ШЗП 13 мм"},  // str|int|float|bool values
  "price": 3299.0,                   // optional, ≥ 0
  "currency": "UAH",                 // default "UAH"
  "in_stock": true                   // default true
}
```

### Search request (`SearchRequest`)

```jsonc
{
  "query": "ударний дриль 600 Вт",   // required, 1–512 chars
  "limit": 10,                        // 1–100, default 10
  "offset": 0,
  "rerank": false,                    // cross-encoder rerank of the hybrid branch
  "mode": "relaxed",                  // "relaxed" (default) | "strict" — see Search modes
  "filters": {                        // all optional, AND-combined
    "brand": "Bosch",                 // case-insensitive
    "category": "Електроінструмент",  // exact match
    "price_min": 1000,
    "price_max": 5000,
    "in_stock": true,
    "attributes": {"Патрон": "ШЗП 13 мм"}   // exact match per key
  }
}
```

Filters are applied inside the vector query (indexed payload fields) for the hybrid branch and in-app for code-branch hits.

### Search response (`SearchResponse`)

```jsonc
{
  "query_kind": "text",              // "code_only" | "mixed" | "text"
  "took_ms": 42.1,
  "total": 7,                         // matches found before offset/limit slicing
  "items": [
    {
      "product": { /* full stored payload, incl. *_norm fields, updated_at, embed_model */ },
      "score": 0.87,                  // branch-dependent — NOT comparable across branches (see note)
      "match": {
        "branch": "hybrid",           // exact | exact_normalized | ean_corrected | fuzzy | hybrid
        "matched_field": null,        // article | product_code | ean13 (code branches)
        "code_score": null,           // 0.80–1.00 for code branches
        "reranked": false,
        "query_coverage": 1.0,        // share of significant query terms found in the product
        "missing_terms": null         // query terms the product lacks (why it's an alternative)
      }
    }
  ],
  "alternatives": []                  // strict mode: ranked near-misses; [] in relaxed mode
}
```

> **`score` is branch-dependent and not comparable across hits.** Hybrid hits carry a Qdrant **RRF** score (typically ~0.01–0.03), code-branch hits carry a **code score** (0.80–1.00), and a reranked hybrid hit carries a raw **cross-encoder logit** (unbounded, can be negative). `items` is returned in final rank order (exact code hits pinned first, then fuzzy code hits interleaved with hybrid results) — treat `score` as a per-hit diagnostic and **do not re-sort by it**, or you will destroy the intended ordering.

### Search modes

Vector search always returns the *nearest* products — even when nothing in the catalog satisfies every requested characteristic. `mode` controls what happens then:

- **`relaxed`** (default) — one ranked list, nearest-first. Hybrid hits still carry `match.query_coverage` and `match.missing_terms`, so clients can see how well each hit matches.
- **`strict`** — `items` contains only products the service is *sure* about: every requirement of the query is present in the product's own fields, or the product was hit by a strong code tier. Everything else that still covers ≥ 50 % of the request lands in `alternatives`, each with `missing_terms` naming exactly what it lacks.

Example: `{"query": "мат плата з hdmi на 1200", "mode": "strict"}` against a catalog where no board has both HDMI and LGA 1200 → `items` is empty (nothing to over-promise), and the LGA 1200 board without HDMI comes back in `alternatives` with `"missing_terms": ["hdmi"]`.

**Requirement extraction.** With `QUERY_LLM_ENABLED=true`, strict mode sends the query (once per unique query, LRU-cached) to the configured LLM backend (`LLM_PROVIDER`: local Ollama or OpenAI-compatible) that generates lexical variants per query token — synonyms, uk/ru/en translations, abbreviations, value formats — with no hand-maintained dictionaries: «на 1200» is covered by `LGA 1200`. If the LLM is off, times out, or answers garbage, the service degrades to per-token heuristics (this is also the relaxed-mode annotation path — relaxed never pays LLM latency). Full analysis, incl. why SPLADE was rejected for uk/ru: `docs/search_semantics.md`.

Variant matching is heuristic and unit-tested: numbers match on digit boundaries (`1200` ≠ `12000`, but matches `LGA1200`), alphabetic tokens match by prefix (`мат` covers «Материнська»), tokens ≥ 5 chars tolerate a changed final char (`плати` covers «плата»).

---

## Request / response examples

### Upsert a product

```bash
curl -X POST http://localhost:8000/api/v1/products \
  -H "X-API-Key: change-me-secret-key" -H "Content-Type: application/json" \
  -d '{
    "external_id": "tool-001",
    "name": "Дриль ударний Bosch GSB 13 RE",
    "brand": "Bosch",
    "category": "Електроінструмент",
    "article": "GSB-13-RE",
    "product_code": "060114E600",
    "ean13": "4006381333931",
    "attributes": {"Потужність": "600 Вт", "Патрон": "ШЗП 13 мм"},
    "price": 3299.0
  }'
```

```json
{"total": 1, "succeeded": 1, "failed": 0, "items": [{"external_id": "tool-001", "ok": true, "error": null}]}
```

### Exact article — answered from the code index in ~3 ms

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "X-API-Key: change-me-secret-key" -H "Content-Type: application/json" \
  -d '{"query": "GSB-13-RE", "limit": 5}'
```

```json
{
  "query_kind": "code_only",
  "took_ms": 3.2,
  "total": 1,
  "items": [{
    "product": {"external_id": "tool-001", "name": "Дриль ударний Bosch GSB 13 RE", "article": "GSB-13-RE", "...": "..."},
    "score": 1.0,
    "match": {"branch": "exact", "matched_field": "article", "code_score": 1.0, "reranked": false}
  }]
}
```

### Mistyped EAN-13 — checksum-aware correction

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "X-API-Key: change-me-secret-key" -H "Content-Type: application/json" \
  -d '{"query": "4006381333932"}'
```

```json
{
  "query_kind": "code_only",
  "took_ms": 2.9,
  "items": [{
    "product": {"external_id": "tool-001", "name": "Дриль ударний Bosch GSB 13 RE", "...": "..."},
    "score": 0.95,
    "match": {"branch": "ean_corrected", "matched_field": "ean13", "code_score": 0.95, "reranked": false}
  }]
}
```

### Mixed query — code hit pinned above semantic results

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "X-API-Key: change-me-secret-key" -H "Content-Type: application/json" \
  -d '{"query": "дриль GSB13RE з кейсом", "limit": 3}'
```

```json
{
  "query_kind": "mixed",
  "took_ms": 186.1,
  "items": [
    {"product": {"name": "Дриль ударний Bosch GSB 13 RE", "...": "..."}, "score": 1.0,
     "match": {"branch": "exact", "matched_field": "article", "code_score": 1.0, "reranked": false}},
    {"product": {"name": "Шуруповерт акумуляторний Makita DDF484Z", "...": "..."}, "score": 0.44,
     "match": {"branch": "hybrid", "matched_field": null, "code_score": null, "reranked": false}}
  ]
}
```

### File import

```bash
curl -X POST http://localhost:8000/api/v1/imports \
  -H "X-API-Key: change-me-secret-key" \
  -F "file=@catalog.csv" \
  -F 'column_mapping={"Артикул виробника": "article", "Штрих-код": "ean13"}'
# → 202 {"job_id": "3f2a…", "status": "pending", ...}

curl -H "X-API-Key: change-me-secret-key" http://localhost:8000/api/v1/imports/3f2a…
# → {"status": "completed", "total": 5000, "processed": 5000, "failed": 3, "errors": [...]}
```

---

## Deployment

### Docker Compose (recommended)

```bash
cp .env.example .env        # set API_KEYS at minimum
docker compose up -d --build
```

Two services:

- **`qdrant`** — `qdrant/qdrant:v1.17.1`, storage in the `qdrant_data` volume, dashboard at http://localhost:6333/dashboard.
- **`api`** — this service, port 8000. Embedding models (~2.2 GB) are downloaded **once** on first start into the `model_cache` volume (`FASTEMBED_CACHE_PATH=/models`, `HF_HOME=/models/hf`); subsequent starts take seconds. Healthcheck allows up to ~5 min of start time for the first download.

Wait for readiness, then run the demo:

```bash
curl http://localhost:8000/ready
# {"status":"ready","indexed_code_points":38}

uv run python scripts/smoke_search.py   # ingests data/sample_products.json + runs every query type
```

The Docker image is a multi-stage build: dependencies resolved by `uv sync --frozen` from `uv.lock`, runtime is `python:3.13-slim` running as a non-root user.

### First catalog load

CPU embedding throughput for `multilingual-e5-large` is roughly 10–30 products/s — a 100k catalog takes 1–3 hours **once**. Options:

- push in batches of ≤1000 via `POST /api/v1/products:batch` (the observed rate: 38 products ≈ 14 s cold),
- or upload a single CSV/XLSX/JSON via `POST /api/v1/imports` and poll the job.

Pre-warm the model cache without starting the API: `uv run python scripts/download_models.py`.

### Scaling notes

- ≤100k products fit comfortably on a single Qdrant node with vectors in RAM (~410 MB dense); no quantization needed. Binary/scalar quantization and Qdrant clustering are the escalation path beyond ~1M.
- The API is stateless **except** for the in-memory CodeIndex (rebuilt from a payload-only scroll at startup, updated incrementally) and the background JobStore. Multiple replicas each hold their own CodeIndex copy — fine at this scale; ingest through one replica or rebuild others periodically if you shard writes.
- **Background jobs are durable** when `JOBS_DB_PATH` is set (SQLite, on a volume in the shipped compose): import and async-batch job state survives a restart, and a job left `running` by a process that died is reaped to `failed` on the next boot (rather than dangling forever). The store is still per-instance, so job polling must hit the replica that owns the DB file; a shared DB / Redis is the multi-replica escalation.

---

## Configuration

All settings via environment / `.env` (see `.env.example`, parsed by pydantic-settings — `app/core/config.py`):

| Variable | Default | Meaning |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `QDRANT_API_KEY` | — | Qdrant API key (if secured) |
| `COLLECTION_NAME` | `products` | main collection |
| `API_KEYS` | — | comma-separated API keys; **empty disables auth** (forbidden when `ENVIRONMENT=production`) |
| `ENVIRONMENT` | `development` | `development` / `staging` / `production`; production refuses to start without `API_KEYS` |
| `LOG_FORMAT` | env-dependent | `text` (development default) or `json` (staging/production default) |
| `METRICS_ENABLED` | `true` | Prometheus metrics at `GET /metrics` |
| `SENTRY_DSN` | — | Sentry error tracking; empty = disabled |
| `SENTRY_TRACES_SAMPLE_RATE` | `0.0` | Sentry performance tracing sample rate (0–1) |
| `DENSE_MODEL` | `intfloat/multilingual-e5-large` | FastEmbed dense model |
| `DENSE_DIM` | `1024` | must match the model |
| `SPARSE_MODEL` | `Qdrant/bm25` | FastEmbed sparse model |
| `SPARSE_LANGUAGE` | `russian` | BM25 stemmer language |
| `EMBED_BATCH_SIZE` / `UPSERT_BATCH_SIZE` | `64` / `256` | pipeline batching |
| `PREFETCH_LIMIT` | `50` | per-branch candidate pool for RRF |
| `RERANK_ENABLED` | `false` | allow `rerank=true` requests |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | cross-encoder model |
| `RERANK_TOP_K` | `50` | how many hybrid hits get reranked |
| `LLM_PROVIDER` | `ollama` | LLM backend: `ollama` (local) or `openai` (OpenAI / any OpenAI-compatible endpoint) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint (`host.docker.internal` from Docker) |
| `OPENAI_API_KEY` | — | required when `LLM_PROVIDER=openai` and an LLM feature is on |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible base URL (vLLM, LiteLLM, Groq, …) |
| `QUERY_LLM_ENABLED` | `false` | LLM query understanding for strict mode |
| `QUERY_LLM_MODEL` | `gemma4:e4b` | model for requirement extraction (`gpt-4o-mini` for openai) |
| `QUERY_LLM_TIMEOUT_S` | `30.0` | LLM call budget; on timeout search degrades to heuristics |
| `INGEST_LLM_ENABLED` | `false` | LLM enrichment at ingest (pre-embedding) |
| `INGEST_LLM_MODEL` | `gemma4:e4b` | model for ingest enrichment |
| `INGEST_LLM_TIMEOUT_S` / `INGEST_LLM_CONCURRENCY` | `60` / `2` | per-product budget and parallelism |
| `INGEST_ENRICH_WITH_ATTRIBUTES` | `true` | enrich products that already have attributes too (for synonyms); `false` = attribute-less only |
| `DEBUG` | `false` | debug logging |

Changing `DENSE_MODEL`/`DENSE_DIM`/`SPARSE_MODEL` against an existing collection triggers the `service_meta` guard: the service exits with a clear reindex instruction instead of mixing incompatible vectors.

### Pre-embedding enrichment (optional)

With `INGEST_LLM_ENABLED=true`, **every** product is enriched once at ingest by the configured LLM: structured `attributes` (filterable facts, gaps only — supplier attributes win on merge), uk/ru/en `synonyms` (into the BM25 sparse text — «материнка» finds "motherboard"), a literal `spec_summary` and `use_cases` (into the dense vector). The synonyms are the highest-value part: a 3-way benchmark (branch `experiment/bge-m3`) found **e5+BM25+enrichment beats raw BGE-M3** on uk/ru/en queries precisely because of them, so enrichment now runs even for products that already have attributes (set `INGEST_ENRICH_WITH_ATTRIBUTES=false` for attribute-less only). Any LLM failure ingests the product unenriched; `content_hash` skips re-enriching unchanged products on re-push. The enrichment block is stored in the payload and counts as product text for strict-mode coverage. Rationale, guardrails and throughput math: `docs/pipeline_assessment.md`.

### Observability

- **Request IDs** — every request gets an `X-Request-ID` (accepted from the client or generated), echoed in the response and stamped on every log record.
- **Logs** — one uniform stream (app + uvicorn) via the root logger: human-readable text in development, single-line JSON in staging/production (`LOG_FORMAT` overrides). `extra={...}` fields become top-level JSON keys.
- **Metrics** — `GET /metrics` (Prometheus): standard HTTP metrics (latency histograms, status codes, in-flight) plus domain metrics `search_requests_total{query_kind}` and `search_latency_seconds{query_kind}`.
- **Errors** — unhandled exceptions are logged with the request id, returned as JSON `500 {"detail": "Internal server error"}`, and captured by Sentry when `SENTRY_DSN` is set (environment tag = `ENVIRONMENT`, no PII sent).

### Optional reranking

```bash
uv sync --group rerank      # sentence-transformers + torch (CPU) — heavy, hence optional
# .env: RERANK_ENABLED=true
# request: {"query": "...", "rerank": true}
```

Best quality on long/ambiguous phrases, +50–200 ms when enabled. Requesting `rerank=true` while disabled returns `400`.

---

## Development

```bash
uv sync --group dev
uv run pytest -q                            # 149 unit tests — no Qdrant/models needed
uv run pytest -m integration -q -o addopts="" # e2e tests against a running compose stack
uv run ruff check app tests scripts         # lint
uv run ruff format app tests scripts        # format
uv run mypy                                 # type check (app/)
uv run pre-commit install                   # ruff + mypy + hygiene checks on every commit
uv run python scripts/gen_sample_data.py    # regenerate data/sample_products.json (valid EAN-13s)
```

CI (GitHub Actions, `.github/workflows/ci.yml`) runs lint, format check, mypy, unit tests, and a Docker image build on every push/PR. See `docs/production_readiness.md` for the production checklist.

### Search-quality evaluation

Relevance is measured, not eyeballed: `eval/queries.jsonl` is a labelled query set (graded 0-3, uk/ru/en × code/text/mixed/strict) and `scripts/eval_search.py` scores the live service with **nDCG@k / Recall@k / MRR, reported per segment** (so you see *where* quality leaks, not just an average). Run it before/after any change that touches retrieval, fusion, enrichment, or ranking.

```bash
# reproducible corpus (ingest sample + reconcile), score, save the baseline:
uv run --no-sync python scripts/eval_search.py --setup --out eval/baseline.json --label prod
# after a change, diff against the baseline (per-segment deltas + per-query regressions):
uv run --no-sync python scripts/eval_search.py --compare eval/baseline.json --label candidate
```

Details, metrics and caveats (small-sample noise, judged-pool bias): `eval/README.md`.

### Console (Streamlit)

`docker compose up -d --build` also starts an internal admin/testing/eval console at **http://localhost:8501** — pages for **Ingest** (form / bulk / price / lifecycle), **Search** (query playground with the full match explanation), and **Evaluate** (run the harness, per-segment scorecard, diff vs baseline, add labels). It talks to the API over HTTP only, so the API key stays server-side. See `ui/README.md`.

Layout:

```
app/
├── main.py            # app factory + lifespan (model warmup, collection init, CodeIndex bootstrap)
├── core/              # settings (pydantic-settings), X-API-Key auth, logging
├── models/            # Pydantic v2 schemas: product, search, imports
├── api/v1/            # routers: products, search, imports, health
└── services/          # the engine (see Components table above)
scripts/               # gen_sample_data, download_models, smoke_search, eval_search
tests/unit             # normalization, EAN, code index, query classification, sync, enrichment
tests/integration      # full e2e over HTTP (marker: integration)
eval/                  # labelled query set + baseline for search-quality scoring
ui/                    # Streamlit admin/testing/eval console (compose service `ui`, :8501)
```

Unit tests never import ONNX (the FastEmbed import is deferred into `EmbeddingService.__init__`), so they run anywhere in ~2 s.

---

## Design decisions & limitations

- **App-side fuzzy index instead of Qdrant trigram vectors.** At ≤100k products × 3 code fields, RapidFuzz's C++ batch scan finishes in tens of milliseconds, models edit distance directly (trigram overlap does not), and keeps the collection schema to two vectors. Trade-off: per-replica memory copy + startup scroll.
- **RRF over weighted score blending.** Rank fusion needs no per-domain weight tuning and is computed server-side by Qdrant in one round trip.
- **Codes excluded from the dense vector, included in the sparse one.** Semantic vectors stay clean; lexical code matching still works inside the hybrid branch.
- **Ukrainian stemming** does not exist in FastEmbed's BM25; the Russian stemmer + verbatim token matching is used, and the dense model carries most of the Ukrainian semantics through RRF.
- **`bge-m3` is not shipped by FastEmbed 0.8** — `multilingual-e5-large` is the strongest multilingual dense model in its catalog and is used by default; the model-profile abstraction makes swapping trivial (with a mandatory reindex, enforced by `service_meta`).
- **SQLite-backed JobStore** (`JOBS_DB_PATH`) — import/async-batch job state is durable across restarts and orphaned `running` jobs are reaped on boot; still per-instance (polling is replica-sticky), so move to a shared DB/Redis if you scale the API horizontally.
- **Windows host note:** the ONNX embedding stack is flaky on bare Windows (MSVC runtime conflicts); the supported runtime is Docker. Unit tests are unaffected.
