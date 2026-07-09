# Product Search Console (Streamlit)

Internal admin / testing / eval UI. Talks to the API over HTTP only (no direct DB access),
so the API key stays server-side and never reaches the browser.

Pages:
- **Home** — catalog stats + readiness.
- **Ingest** — single-product form, bulk JSON upload, price/availability, archive/delete/reconcile.
- **Search** — query playground with mode / filters and the full match explanation (branch,
  coverage, missing terms, score, took_ms).
- **Evaluate** — run the labelled eval harness against the live service, nDCG/Recall/MRR per
  segment, diff vs `eval/baseline.json`, and append new labels to `eval/queries.jsonl`.

## Run

With Docker (recommended — one command brings up qdrant + api + ui):

```bash
docker compose up -d --build        # UI at http://localhost:8501
```

Locally against a running API:

```bash
uv sync --group ui
API_BASE_URL=http://localhost:8000 API_KEY=change-me-secret-key \
  uv run --group ui streamlit run ui/Home.py
```

Config via env: `API_BASE_URL` (default `http://localhost:8000`), `API_KEY` (first of
`API_KEYS` if comma-separated).
