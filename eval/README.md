# Search evaluation harness

Turns "is search good?" into numbers you can compare across changes — the prerequisite
for any principled ranking/model/fusion decision (and a CI regression guard).

## Files

- `queries.jsonl` — the labelled query set. One JSON object per line:
  ```json
  {"query": "...", "lang": "uk|ru|en", "kind": "code|text|mixed",
   "mode": "relaxed|strict", "relevant": {"external_id": grade}}
  ```
  Grades: `3` exact / `2` clearly relevant / `1` related-or-accessory / (absent = irrelevant).
  This is a hand-authored **seed** (48 queries over the sample catalog) — extend it as real
  query logs arrive; labels key on `external_id`, so they survive catalog content changes.
- `baseline.json` — a committed reference run to diff future changes against.
- `../scripts/eval_search.py` — the runner.

## Use

```bash
# reproducible corpus (ingest sample catalog + reconcile so nothing else is searchable),
# score the live service, save the baseline:
uv run --no-sync python scripts/eval_search.py --setup --label prod --out eval/baseline.json

# after a change, score again and diff against the baseline (shows per-segment deltas
# and per-query nDCG regressions):
uv run --no-sync python scripts/eval_search.py --compare eval/baseline.json --label candidate

# optional: LLM-as-judge the pooled top-k the gold labels did NOT mark — surfaces
# relevants the seed missed (mitigates judged-pool bias). Costs one LLM call per pooled hit.
uv run --no-sync python scripts/eval_search.py --judge
```

Metrics: **nDCG@k** (graded, position-weighted — the ranking-quality gold standard),
**Recall@k**, **MRR** — reported overall and per segment (`lang`, `kind`, `mode`).

## Why segments matter

The aggregate hides where quality actually leaks. Read the per-segment lines: e.g. the
seed baseline scores ~1.0 on code/mixed and relaxed text but much lower on `mode:strict`
— which is where to spend effort next. Always judge a change per-segment, not just overall.

## Caveats (honest)

- **Small sample** — 48 queries is a seed; a single query swings a segment. Grow it before
  trusting small deltas.
- **Judged-pool bias** — the runner scores against *labelled* products only; a genuinely
  better system that surfaces an unlabelled-but-relevant product looks worse until you add
  that label. Use `--judge` (or pool several systems) periodically to find missing labels.
- **Labels are a seed**, hand-authored from the sample catalog — not real user judgments.
  Replace/augment with click-derived labels once impression/click logging exists.
