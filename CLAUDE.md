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
- Observability is wired in `app/core/{logging,monitoring}.py` + `app/main.py`: request-id on every log record, JSON logs outside development, Prometheus `/metrics`, Sentry via `SENTRY_DSN`.

## This Windows host (important)

- The ONNX embedding stack is broken on bare Windows here; **the runtime for the embedding stack is Docker**. Unit tests are unaffected.
- After **every** `uv sync`, re-copy MSVC DLLs into onnxruntime (see auto-memory `onnxruntime-dll-fix-windows`); prefer `uv run --no-sync` afterwards.
- `uv` is winget-installed; refresh PATH per PowerShell call:
  `$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")`
- An intermittent onnxruntime access-violation trace during pytest collection is a known flake — ignore it if tests pass.
- E2E tests default to `X-API-Key: change-me-secret-key` (`E2E_API_KEY` overrides); local `.env` is gitignored.
