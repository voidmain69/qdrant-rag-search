"""Manual E2E demo against a running stack: ingest sample products, run every query type.

Usage:
    uv run python scripts/smoke_search.py [--base-url http://localhost:8000] [--api-key change-me-secret-key]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

QUERIES = [
    ("Семантика (фраза)", {"query": "бездротовий пилосос для дому"}),
    ("Характеристики (гібрид)", {"query": "акумулятор 18V 5Ah Li-Ion"}),
    ("Точний артикул", {"query": "GSB-13-RE"}),
    ("Артикул з одруком", {"query": "GSB-13-RF"}),
    ("Артикул кирилицею (гомогліфи)", {"query": "ВL1850В"}),
    ("Точний EAN-13", {"query": "4006381333931"}),
    ("EAN-13 з помилковою цифрою", {"query": "4006381333932"}),
    ("Змішаний запит", {"query": "дриль GSB13RE з кейсом"}),
    ("З фільтром бренду", {"query": "пилосос", "filters": {"brand": "Samsung"}}),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default="change-me-secret-key")
    parser.add_argument("--skip-ingest", action="store_true")
    args = parser.parse_args()

    client = httpx.Client(base_url=args.base_url, headers={"X-API-Key": args.api_key}, timeout=120)

    ready = client.get("/ready")
    if ready.status_code != 200:
        print(f"Service not ready: {ready.status_code} {ready.text}")
        sys.exit(1)
    print(f"Ready: {ready.json()}")

    if not args.skip_ingest:
        data = json.loads(
            (Path(__file__).resolve().parents[1] / "data" / "sample_products.json").read_text("utf-8")
        )
        started = time.perf_counter()
        resp = client.post("/api/v1/products:batch", json={"items": data})
        resp.raise_for_status()
        result = resp.json()
        print(
            f"Ingested {result['succeeded']}/{result['total']} products "
            f"in {time.perf_counter() - started:.1f}s"
        )

    for title, body in QUERIES:
        resp = client.post("/api/v1/search", json={**body, "limit": 3})
        resp.raise_for_status()
        data = resp.json()
        print(f"\n=== {title}: {body['query']!r} ({data['query_kind']}, {data['took_ms']} ms)")
        for hit in data["items"]:
            product = hit["product"]
            match = hit["match"]
            print(
                f"  [{match['branch']:>16}] {hit['score']:.4f}  "
                f"{product.get('name')}  (art={product.get('article')})"
            )


if __name__ == "__main__":
    main()
