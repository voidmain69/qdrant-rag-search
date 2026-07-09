"""Evaluate search quality: run the labelled harness against the live service, show
nDCG / Recall / MRR per segment, diff against the committed baseline, and add labels."""

import json
import sys

import client
import streamlit as st

sys.path.insert(0, str(client.REPO_ROOT / "scripts"))
import eval_search

DATASET = client.REPO_ROOT / "eval" / "queries.jsonl"
BASELINE = client.REPO_ROOT / "eval" / "baseline.json"

st.set_page_config(page_title="Evaluate", page_icon="📊", layout="wide")
st.title("📊 Search-quality evaluation")

dataset = eval_search.load_dataset(DATASET)
st.caption(f"{len(dataset)} labelled queries · metrics: nDCG@k / Recall@k / MRR, per segment")

c1, c2, c3 = st.columns([1, 1, 2])
k = c1.number_input("k", 1, 50, 10)
setup = c2.checkbox("reproducible corpus", help="ingest sample catalog + reconcile (mutates the catalog)")
do_diff = c3.checkbox("diff vs committed baseline", value=BASELINE.exists())

if st.button("Run evaluation", type="primary"):
    ev = eval_search.Client(client.BASE_URL, client.API_KEY)
    with st.spinner("scoring…"):
        if setup:
            ev.setup()
        results = eval_search.run(ev, dataset, int(k))

    o = results["overall"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("nDCG", f"{o['ndcg']:.3f}")
    m2.metric("Recall", f"{o['recall']:.3f}")
    m3.metric("MRR", f"{o['mrr']:.3f}")
    m4.metric("queries", o["n"])

    st.subheader("Per segment")
    seg_rows = [
        {
            "segment": seg,
            "nDCG": round(m["ndcg"], 3),
            "Recall": round(m["recall"], 3),
            "MRR": round(m["mrr"], 3),
            "n": m["n"],
        }
        for seg, m in results["segments"].items()
    ]
    st.dataframe(seg_rows, use_container_width=True, hide_index=True)

    st.subheader("Weakest queries (nDCG)")
    worst = sorted(results["per_query"], key=lambda r: r["ndcg"])[:10]
    st.dataframe(
        [
            {
                "nDCG": round(r["ndcg"], 2),
                "lang": r["lang"],
                "kind": r["kind"],
                "query": r["query"],
                "top3": r["ranked"][:3],
            }
            for r in worst
        ],
        use_container_width=True,
        hide_index=True,
    )

    if do_diff and BASELINE.exists():
        baseline = json.loads(BASELINE.read_text("utf-8"))
        st.subheader("Diff vs baseline")
        st.code(eval_search.format_diff(baseline, results), language="text")

st.divider()
st.subheader("Add a labelled query")
with st.form("label"):
    lq = st.text_input("query")
    c1, c2, c3 = st.columns(3)
    lang = c1.selectbox("lang", ["uk", "ru", "en"])
    kind = c2.selectbox("kind", ["text", "code", "mixed"])
    lmode = c3.selectbox("mode", ["relaxed", "strict"])
    rel_raw = st.text_input("relevant (external_id:grade, ...)", placeholder="el-003:3, home-001:2")
    if st.form_submit_button("Append to queries.jsonl") and lq and rel_raw:
        try:
            relevant = {}
            for part in rel_raw.split(","):
                pid, grade = part.split(":")
                relevant[pid.strip()] = int(grade)
            row = {"query": lq, "lang": lang, "kind": kind, "mode": lmode, "relevant": relevant}
            with DATASET.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            st.success("added — re-run evaluation to include it")
        except Exception as exc:
            st.error(f"could not parse: {exc}")
