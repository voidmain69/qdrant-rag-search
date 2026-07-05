# CLAUDE.md

Hybrid product search service on Qdrant (FastAPI, Python 3.13, uv): semantic (dense e5) + BM25 sparse with server-side RRF fusion + in-memory exact/fuzzy product-code matching. No LLM generation. Catalog is multilingual (uk/ru/en). Full architecture, API contract and flows: `README.md`; production checklist: `docs/production_readiness.md`.

## Commands

```bash
uv run pytest -q                              # unit tests (fast, no Qdrant/ONNX)
uv run pytest -m integration -q -o addopts="" # e2e, needs `docker compose up -d --build`
uv run ruff check app tests scripts           # lint
uv run ruff format app tests scripts          # format
uv run mypy                                   # type check (app/)
docker compose up -d --build                  # full stack (requires .env with API_KEYS)
```

All five checks (lint, format --check, mypy, pytest, docker build) must pass — CI (`.github/workflows/ci.yml`) enforces them.

## Code style

- ruff is the source of truth: line length 110, rules `E W F I UP B SIM C4 RUF ASYNC`; `ruff format` for formatting. RUF001–003 (ambiguous Cyrillic) are disabled **deliberately** — Cyrillic homoglyphs are this project's domain, never "fix" Cyrillic/Latin lookalikes in strings or tests.
- mypy runs with `disallow_untyped_defs` on `app/` — annotate every function; heavy optional deps are typed via `if TYPE_CHECKING:` imports (see `app/services/reranker.py`).
- No magic numbers: score tiers and fuzzy weights live as named constants in `app/services/code_index.py` (`EXACT_SCORE`, `STRONG_CODE_SCORE`, …) — import them, don't re-hardcode.
- Docstrings explain *why* (design tradeoffs), not *what*. Module docstrings carry the rationale (see `code_index.py`, `ean.py`).
- Pydantic v2 everywhere; models use `str_strip_whitespace`; validators in `mode="after"`.
- Never import fastembed/ONNX at module level — defer into `__init__`/methods so unit tests stay ONNX-free.
- Sync CPU-bound work goes through `asyncio.to_thread`; never block the event loop.
- pre-commit is configured (`uv run pre-commit install`): ruff, mypy, YAML/TOML hygiene.

## Architecture invariants

- **Score tier ordering**: exact 1.0 > skeleton 0.97 > EAN 0.95 > fuzzy ≤ 0.949. Fuzzy must stay strictly below `STRONG_CODE_SCORE`; sorting by score must reproduce tier order.
- **Dense text excludes codes, sparse text includes them** (`normalization.py`) — do not leak SKU/EAN into the dense embedding text.
- **Ingestion is all-or-nothing** per batch; point ids are `uuid5(namespace, external_id)` — deterministic, idempotent.
- **`service_meta` collection guards model compatibility** — changing DENSE_MODEL/DIM/SPARSE_MODEL requires reindex; the service refuses to start on mismatch.
- **CodeIndex is per-replica in-memory state**, guarded by a lock; every mutation must go through `add_product`/`remove_product`.
- `ENVIRONMENT=production` refuses to start with empty `API_KEYS`; compose has no default secret (`${API_KEYS:?}` — `.env` is mandatory).
- **LLM backend** (`app/services/llm.py`): a single `LLMClient` abstraction — `OllamaClient` or `OpenAIClient` (OpenAI or any OpenAI-compatible endpoint), selected by `LLM_PROVIDER`, built once in `main.py` and shared by both LLM features. Providers are added here; callers stay provider-agnostic and only ever see `complete_json` + `LLMError`.
- **Strict mode & coverage**: `strict` search splits confident hits from `alternatives` via requirement coverage (`app/services/coverage.py`). Synonymization is dictionary-free: LLM query understanding (`app/services/query_understanding.py`, `QUERY_LLM_ENABLED`), strict-mode-only, LRU-cached, always degrades to token heuristics on failure. No hand-maintained synonym dictionaries — see `docs/search_semantics.md` (incl. why SPLADE was rejected: no multilingual checkpoint in fastembed).
- **Ingest enrichment** (`app/services/enrichment.py`, `INGEST_LLM_ENABLED`): LLM enriches **every** product by default (supplier attributes still win on merge) — the win is the uk/ru/en synonyms; `INGEST_ENRICH_WITH_ATTRIBUTES=false` restores "attribute-less only". Failure → ingest unenriched. Synonyms go to sparse text only, spec/use-cases to dense. A 3-way benchmark (branch `experiment/bge-m3`) found e5+BM25+enrichment beats raw BGE-M3 thanks to these synonyms — see `docs/pipeline_assessment.md`.
- **Catalog sync & lifecycle**: three orthogonal levers — `in_stock` (buyable, still searchable), `status=archived` (hidden from default search, retained, reversible), hard `delete` (vectors gone). Search hides `archived` by default via `must_not status=archived` (legacy points without `status` count as active); `include_archived=true` opts in; code branch honours it in `payload_matches_filters`. `upsert` skips embedding when `content_hash` (hash of embedding-affecting fields) is unchanged — `REEMBED_UNCHANGED=true` forces it. `POST /products:reconcile` **archives** orphans (never deletes) with `dry_run` (default) + `max_archived` cap. Sync ops live in `IngestService` (`set_archived`/`delete_products`/`reconcile`/`diff`/`stats`).
- Observability is wired in `app/core/{logging,monitoring}.py` + `app/main.py`: request-id on every log record, JSON logs outside development, Prometheus `/metrics`, Sentry via `SENTRY_DSN`.

## This Windows host (important)

- The ONNX embedding stack is broken on bare Windows here; **the runtime for the embedding stack is Docker**. Unit tests are unaffected.
- After **every** `uv sync`, re-copy MSVC DLLs into onnxruntime (see auto-memory `onnxruntime-dll-fix-windows`); prefer `uv run --no-sync` afterwards.
- `uv` is winget-installed; refresh PATH per PowerShell call:
  `$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")`
- An intermittent onnxruntime access-violation trace during pytest collection is a known flake — ignore it if tests pass.
- E2E tests default to `X-API-Key: change-me-secret-key` (`E2E_API_KEY` overrides); local `.env` is gitignored.
- Ollama runs in the `ollama` container (RTX 3080 Ti, model `gemma4:e4b`); cold model load can take minutes and blocks its HTTP API — warm it up before latency-sensitive tests. From the api container it's `http://host.docker.internal:11434`.
