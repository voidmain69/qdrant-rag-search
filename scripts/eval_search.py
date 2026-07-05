"""Search evaluation harness.

Runs a labelled query set (eval/queries.jsonl) against the live /search API and reports
Recall@k / MRR / nDCG@k overall and per segment (lang, kind, mode) — turning "is search
good?" into numbers you can compare across changes.

Usage:
  # make the corpus reproducible (ingest sample catalog, archive anything else), then score
  uv run --no-sync python scripts/eval_search.py --setup --out eval/baseline.json --label baseline
  # after a change, score again and diff against the saved baseline
  uv run --no-sync python scripts/eval_search.py --compare eval/baseline.json --label candidate
  # optional: LLM-judge the pooled top-k the gold labels missed (judged-pool-bias check)
  uv run --no-sync python scripts/eval_search.py --judge

Labels are a hand-authored seed (grades 0-3); extend eval/queries.jsonl as real query
logs arrive. Console may garble Cyrillic — the full scorecard is written to eval/*.txt.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_DATASET = ROOT / "eval" / "queries.jsonl"


def safe_print(text: str) -> None:
    """Console may be cp1251 (Windows) — never let an encoding error crash the run."""
    sys.stdout.buffer.write(text.encode("utf-8", "replace") + b"\n")
    sys.stdout.flush()


def load_dataset(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def metrics_for(ranked: list[str], relevant: dict[str, int], k: int) -> tuple[float, float, float]:
    topk = ranked[:k]
    rel_ids = {i for i, g in relevant.items() if g >= 1}
    strong_ids = {i for i, g in relevant.items() if g >= 2}

    recall = len([i for i in topk if i in rel_ids]) / len(rel_ids) if rel_ids else 0.0

    mrr = 0.0
    for rank, pid in enumerate(topk):
        if pid in strong_ids:
            mrr = 1.0 / (rank + 1)
            break

    dcg = sum((2 ** relevant.get(pid, 0) - 1) / math.log2(rank + 2) for rank, pid in enumerate(topk))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(rank + 2) for rank, g in enumerate(ideal))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return recall, mrr, ndcg


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class Client:
    def __init__(self, base_url: str, api_key: str):
        self._c = httpx.Client(base_url=base_url, headers={"X-API-Key": api_key}, timeout=120)

    def search(self, query: str, mode: str, k: int) -> tuple[list[str], dict[str, dict]]:
        resp = self._c.post("/api/v1/search", json={"query": query, "limit": k, "mode": mode})
        resp.raise_for_status()
        items = resp.json()["items"]
        ranked = [it["product"]["external_id"] for it in items]
        cards = {it["product"]["external_id"]: it["product"] for it in items}
        return ranked, cards

    def setup(self) -> None:
        """Reproducible corpus: ingest the sample catalog, then reconcile so anything else
        (leftover demo products) is archived and hidden from search."""
        data = json.loads((ROOT / "data" / "sample_products.json").read_text("utf-8"))
        r = self._c.post("/api/v1/products:batch", json={"items": data})
        r.raise_for_status()
        ids = [p["external_id"] for p in data]
        r = self._c.post(
            "/api/v1/products:reconcile",
            json={"external_ids": ids, "dry_run": False, "max_archived": 1_000_000},
        )
        r.raise_for_status()
        print(f"setup: ingested {len(ids)} products, archived {r.json()['archived_count']} others")


def run(client: Client, dataset: list[dict], k: int) -> dict:
    per_query = []
    for row in dataset:
        mode = row.get("mode", "relaxed")
        ranked, _cards = client.search(row["query"], mode, k)
        recall, mrr, ndcg = metrics_for(ranked, row["relevant"], k)
        per_query.append(
            {
                "query": row["query"],
                "lang": row["lang"],
                "kind": row["kind"],
                "mode": mode,
                "recall": recall,
                "mrr": mrr,
                "ndcg": ndcg,
                "ranked": ranked[:k],
            }
        )

    def agg(rows: list[dict]) -> dict:
        return {
            "recall": mean([r["recall"] for r in rows]),
            "mrr": mean([r["mrr"] for r in rows]),
            "ndcg": mean([r["ndcg"] for r in rows]),
            "n": len(rows),
        }

    segments: dict[str, dict] = {}
    for axis in ("lang", "kind", "mode"):
        by = defaultdict(list)
        for r in per_query:
            by[f"{axis}:{r[axis]}"].append(r)
        for seg, rows in sorted(by.items()):
            segments[seg] = agg(rows)

    return {"k": k, "overall": agg(per_query), "segments": segments, "per_query": per_query}


def format_scorecard(results: dict, label: str) -> str:
    out = [f"=== eval scorecard: {label}  (k={results['k']}, n={results['overall']['n']}) ==="]

    def line(name: str, m: dict) -> str:
        return f"  {name:20} nDCG={m['ndcg']:.3f}  Recall={m['recall']:.3f}  MRR={m['mrr']:.3f}  (n={m['n']})"

    out.append(line("OVERALL", results["overall"]))
    out.append("  --- segments ---")
    out.extend(line(seg, m) for seg, m in results["segments"].items())
    out.append("  --- weakest queries (nDCG) ---")
    worst = sorted(results["per_query"], key=lambda r: r["ndcg"])[:8]
    for r in worst:
        out.append(f"    nDCG={r['ndcg']:.2f} [{r['lang']}/{r['kind']}] {r['query']}  ->{r['ranked'][:3]}")
    return "\n".join(out)


def format_diff(baseline: dict, candidate: dict) -> str:
    out = ["=== diff: candidate − baseline ==="]

    def d(seg: str, b: dict, c: dict) -> str:
        return (
            f"  {seg:20} nDCG {c['ndcg']:.3f} ({c['ndcg'] - b['ndcg']:+.3f})  "
            f"Recall {c['recall']:.3f} ({c['recall'] - b['recall']:+.3f})  "
            f"MRR {c['mrr']:.3f} ({c['mrr'] - b['mrr']:+.3f})"
        )

    out.append(d("OVERALL", baseline["overall"], candidate["overall"]))
    for seg in candidate["segments"]:
        if seg in baseline["segments"]:
            out.append(d(seg, baseline["segments"][seg], candidate["segments"][seg]))
    out.append("  --- per-query regressions (nDCG drop) ---")
    base_by_q = {r["query"]: r for r in baseline["per_query"]}
    regressions = []
    for r in candidate["per_query"]:
        b = base_by_q.get(r["query"])
        if b and r["ndcg"] < b["ndcg"] - 1e-9:
            regressions.append((r["ndcg"] - b["ndcg"], r["query"], b["ndcg"], r["ndcg"]))
    for delta, query, bn, cn in sorted(regressions):
        out.append(f"    {delta:+.2f}  {query}  ({bn:.2f} → {cn:.2f})")
    if not regressions:
        out.append("    (none)")
    return "\n".join(out)


def judge_pool(client: Client, dataset: list[dict], k: int) -> str:
    """LLM-as-judge over pooled top-k the gold labels did not mark — surfaces relevants the
    gold set missed (judged-pool bias). Uses the configured LLM backend."""
    from app.core.config import get_settings
    from app.services.llm import LLMError, build_llm_client

    settings = get_settings()
    llm = build_llm_client(settings)
    model = settings.query_llm_model
    prompt_t = (
        "Query: {q}\nProduct: {name}\n"
        "Is this product a relevant search result for the query? "
        'Respond JSON only: {"grade": 0-3} (3=exact, 2=relevant, 1=related, 0=irrelevant).'
    )
    import asyncio

    async def grade(query: str, name: str) -> int:
        try:
            raw = await llm.complete_json(
                prompt_t.replace("{q}", query).replace("{name}", name),
                model=model,
                max_tokens=20,
                timeout=settings.query_llm_timeout_s,
            )
            return int(json.loads(raw).get("grade", 0))
        except (LLMError, ValueError, KeyError):
            return -1

    async def run_judge() -> str:
        lines = ["=== LLM-judge: relevants the gold labels missed (pooled top-k) ==="]
        misses = 0
        for row in dataset:
            ranked, cards = client.search(row["query"], row.get("mode", "relaxed"), k)
            for pid in ranked[:k]:
                if pid not in row["relevant"]:
                    g = await grade(row["query"], str(cards.get(pid, {}).get("name", "")))
                    if g >= 2:
                        misses += 1
                        lines.append(f"    grade={g} [{row['query']}] → {pid} ({cards[pid].get('name')})")
        lines.append(f"  {misses} unlabelled-but-relevant hits — add these to eval/queries.jsonl")
        await llm.aclose()
        return "\n".join(lines)

    return asyncio.run(run_judge())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--api-key", default="change-me-secret-key")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--label", default="run")
    ap.add_argument("--setup", action="store_true", help="ingest sample catalog + reconcile first")
    ap.add_argument("--out", type=Path, help="save results JSON")
    ap.add_argument("--compare", type=Path, help="baseline results JSON to diff against")
    ap.add_argument("--judge", action="store_true", help="LLM-judge pooled top-k (bias check)")
    args = ap.parse_args()

    client = Client(args.base_url, args.api_key)
    dataset = load_dataset(args.dataset)

    if args.setup:
        client.setup()

    results = run(client, dataset, args.k)
    results["label"] = args.label

    # write files BEFORE printing so a cp1251 console can't lose the results
    scorecard = format_scorecard(results, args.label)
    (ROOT / "eval" / f"scorecard-{args.label}.txt").write_text(scorecard, encoding="utf-8")
    if args.out:
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    safe_print(scorecard)

    if args.compare:
        baseline = json.loads(args.compare.read_text("utf-8"))
        diff = format_diff(baseline, results)
        (ROOT / "eval" / f"diff-{args.label}.txt").write_text(diff, encoding="utf-8")
        safe_print("\n" + diff)

    if args.judge:
        report = judge_pool(client, dataset, args.k)
        (ROOT / "eval" / "judge-report.txt").write_text(report, encoding="utf-8")
        safe_print("\n" + report)


if __name__ == "__main__":
    main()
