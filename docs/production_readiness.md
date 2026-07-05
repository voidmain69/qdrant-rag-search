# Production readiness checklist

Status legend: ✅ done · 🟡 partial / documented tradeoff · ⬜ deliberately out of scope (do at the infra layer or when scaling).

## Correctness & API contract

| Item | Status | Notes |
|---|---|---|
| Idempotent ingestion (deterministic point ids, upsert semantics) | ✅ | `uuid5(namespace, external_id)`; PUT/POST converge on one path |
| Input validation on every endpoint | ✅ | Pydantic v2; whitespace stripped (`str_strip_whitespace`), EAN digits-only, size limits on all fields |
| Consistent error contract | ✅ | 401/400/404/413/422 + JSON 500 from the global exception handler; `rerank=true` on a non-rerank instance is 400 for **every** query kind |
| Embedding-model / collection compatibility guard | ✅ | `service_meta` collection; refuses to start on mismatch |
| Pagination semantics | 🟡 | `total` counts candidates within the prefetch window (documented), not the whole collection |

## Security

| Item | Status | Notes |
|---|---|---|
| API key auth, constant-time comparison | ✅ | `X-API-Key`, `hmac.compare_digest` |
| No default secrets in deployment configs | ✅ | compose fails fast if `API_KEYS` is unset (`${API_KEYS:?...}`) |
| Refuse unauthenticated production start | ✅ | `ENVIRONMENT=production` + empty `API_KEYS` → startup error |
| Non-root container | ✅ | `appuser` in Dockerfile |
| Upload size limit | ✅ | 100 MB on `/imports` |
| No stack traces / internals in responses | ✅ | generic JSON 500; details go to logs + Sentry only |
| TLS termination, rate limiting, request body cap | ⬜ | at the reverse proxy / ingress (nginx `client_max_body_size`, etc.) |

## Reliability

| Item | Status | Notes |
|---|---|---|
| Liveness vs readiness split | ✅ | `/health` (process up) vs `/ready` (startup done **and** Qdrant reachable — live ping) |
| Fail-fast startup | ✅ | model load, collection init and CodeIndex bootstrap complete before `ready=true` |
| Bounded in-memory state | ✅ | JobStore evicts finished jobs beyond 500; CodeIndex is by design per-replica (README scaling notes) |
| Thread-safety of shared state | ✅ | CodeIndex guards reads and mutations with a lock; fuzzy scan works on snapshots |
| Graceful shutdown | 🟡 | uvicorn drains requests; an in-flight import job dies with the process (in-memory JobStore — documented single-instance tradeoff, move to Redis/DB when scaling out) |
| Multi-replica story | 🟡 | stateless except CodeIndex + JobStore; documented in README scaling notes |

## Observability

| Item | Status | Notes |
|---|---|---|
| Structured logs | ✅ | JSON lines in staging/production, text in development; uvicorn logs unified into the same stream |
| Request correlation | ✅ | `X-Request-ID` middleware; id on every log record and response |
| Metrics | ✅ | Prometheus `/metrics`: HTTP latency/status + `search_requests_total{query_kind}`, `search_latency_seconds{query_kind}` |
| Error tracking | ✅ | Sentry via `SENTRY_DSN` (off by default, no PII), environment-tagged |
| Business event logging | ✅ | `search completed` event with query_kind / took_ms / total / rerank |
| Dashboards & alerts | ⬜ | wire `/metrics` into your Prometheus + Grafana / Alertmanager |

## Delivery & code quality

| Item | Status | Notes |
|---|---|---|
| Lint + format enforced | ✅ | ruff check (`E W F I UP B SIM C4 RUF ASYNC`) + ruff format; homoglyph rules (RUF001-003) disabled — Cyrillic lookalikes are the domain |
| Static typing | ✅ | mypy (`disallow_untyped_defs` on `app/`), clean |
| Tests | ✅ | 62 unit (no Qdrant/ONNX needed) + 13 e2e against the compose stack |
| CI | ✅ | GitHub Actions: lint, format check, mypy, unit tests, Docker build |
| Pre-commit hooks | ✅ | ruff, ruff-format, mypy, YAML/TOML checks |
| Reproducible builds | ✅ | `uv.lock` + `uv sync --frozen` in a multi-stage Docker build |
| Dependency pinning ranges | ✅ | upper bounds on fastapi/qdrant-client/fastembed/rapidfuzz |

## Data

| Item | Status | Notes |
|---|---|---|
| Qdrant persistence | ✅ | named volume `qdrant_data` |
| Backups | ⬜ | schedule Qdrant snapshots (`POST /collections/{name}/snapshots`) off-host |
| Reindex procedure | ✅ | documented (drop `products` + `service_meta`, restart, re-ingest) |
