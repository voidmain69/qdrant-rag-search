"""Thin HTTP client over the search API for the Streamlit console.

The API key lives here (server-side, in the Streamlit process) — never in the browser.
API_KEYS may be comma-separated; we use the first."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
# No fallback secret: the console wields full write/delete power, so a missing key must
# fail loudly rather than silently ship a well-known default. API_KEYS may be
# comma-separated; the console uses the first.
_raw_key = os.environ.get("API_KEY", "").split(",")[0].strip()
if not _raw_key:
    raise RuntimeError("API_KEY is not set — the console has no key to authenticate to the API")
API_KEY = _raw_key


def _client(timeout: float = 60) -> httpx.Client:
    return httpx.Client(base_url=BASE_URL, headers={"X-API-Key": API_KEY}, timeout=timeout)


def _unwrap(resp: httpx.Response) -> Any:
    if resp.is_success:
        return resp.json()
    try:
        detail = resp.json().get("detail")
    except Exception:
        detail = resp.text
    raise RuntimeError(f"HTTP {resp.status_code}: {detail}")


# --- read ---


def ready() -> tuple[int, dict]:
    try:
        with _client(10) as c:
            r = c.get("/ready")
            return r.status_code, (r.json() if r.content else {})
    except httpx.HTTPError as exc:
        return 0, {"error": str(exc)}


def stats() -> dict:
    with _client() as c:
        return _unwrap(c.get("/api/v1/products/stats"))


def search(payload: dict) -> dict:
    with _client(180) as c:
        return _unwrap(c.post("/api/v1/search", json=payload))


# --- write ---


def upsert(product: dict) -> dict:
    with _client(180) as c:
        return _unwrap(c.post("/api/v1/products", json=product))


def upsert_batch(items: list[dict]) -> dict:
    with _client(600) as c:
        return _unwrap(c.post("/api/v1/products:batch", json={"items": items}))


def update_price(external_id: str, payload: dict) -> dict:
    with _client() as c:
        return _unwrap(c.patch(f"/api/v1/products/{external_id}/price", json=payload))


def archive(external_ids: list[str], archived: bool) -> dict:
    with _client() as c:
        return _unwrap(
            c.post("/api/v1/products:archive", json={"external_ids": external_ids, "archived": archived})
        )


def delete(external_ids: list[str]) -> dict:
    with _client() as c:
        return _unwrap(c.post("/api/v1/products:delete", json={"external_ids": external_ids}))


def reconcile(external_ids: list[str], dry_run: bool, max_archived: int | None) -> dict:
    body: dict = {"external_ids": external_ids, "dry_run": dry_run}
    if max_archived is not None:
        body["max_archived"] = max_archived
    with _client(120) as c:
        return _unwrap(c.post("/api/v1/products:reconcile", json=body))
