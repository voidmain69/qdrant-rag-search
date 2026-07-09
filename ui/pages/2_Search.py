"""Search playground: query + mode + filters, with the full match explanation
(branch, coverage, missing terms, score) so you can see WHY each result ranked."""

import json

import client
import streamlit as st

st.set_page_config(page_title="Search", page_icon="🔎", layout="wide")
st.title("🔎 Search playground")

c1, c2, c3, c4 = st.columns([4, 1.2, 1, 1])
query = c1.text_input("query", value="бездротовий пилосос для дому")
mode = c2.selectbox("mode", ["relaxed", "strict"])
limit = c3.number_input("limit", 1, 100, 10)
rerank = c4.checkbox("rerank")
include_archived = st.checkbox("include archived")

with st.expander("filters"):
    f1, f2, f3 = st.columns(3)
    brand = f1.text_input("brand")
    category = f2.text_input("category")
    in_stock = f3.selectbox("in_stock", ["(any)", "true", "false"])
    price_min = f1.number_input("price_min", 0.0, value=0.0, step=1.0)
    price_max = f2.number_input("price_max (0 = none)", 0.0, value=0.0, step=1.0)
    attrs_raw = st.text_area("attributes (JSON object)", value="{}")


def build_filters() -> dict | None:
    flt: dict = {}
    if brand:
        flt["brand"] = brand
    if category:
        flt["category"] = category
    if in_stock != "(any)":
        flt["in_stock"] = in_stock == "true"
    if price_min:
        flt["price_min"] = price_min
    if price_max:
        flt["price_max"] = price_max
    try:
        attrs = json.loads(attrs_raw or "{}")
        if attrs:
            flt["attributes"] = attrs
    except json.JSONDecodeError:
        st.warning("attributes is not valid JSON — ignored")
    return flt or None


def render_hits(hits: list[dict]) -> None:
    rows = []
    for h in hits:
        p, m = h["product"], h["match"]
        rows.append(
            {
                "external_id": p.get("external_id"),
                "name": p.get("name"),
                "brand": p.get("brand"),
                "price": p.get("price"),
                "in_stock": p.get("in_stock"),
                "score": round(h["score"], 4),
                "branch": m.get("branch"),
                "coverage": m.get("query_coverage"),
                "missing": ", ".join(m.get("missing_terms") or []),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


if st.button("Search", type="primary") and query.strip():
    payload = {
        "query": query,
        "mode": mode,
        "limit": int(limit),
        "rerank": rerank,
        "include_archived": include_archived,
        "filters": build_filters(),
    }
    try:
        data = client.search(payload)
    except Exception as exc:
        st.error(str(exc))
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("query_kind", data["query_kind"])
        m2.metric("took_ms", data["took_ms"])
        m3.metric("total", data["total"])
        m4.metric("alternatives", len(data.get("alternatives", [])))

        st.subheader(f"Items ({len(data['items'])})")
        if data["items"]:
            render_hits(data["items"])
        else:
            st.info("no confident items")

        if data.get("alternatives"):
            st.subheader(f"Alternatives ({len(data['alternatives'])}) — near-misses")
            render_hits(data["alternatives"])

        with st.expander("raw response JSON"):
            st.json(data)
