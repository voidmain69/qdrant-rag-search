"""Ingest & manage the catalog: single product form, bulk upload, price/availability,
lifecycle (archive / delete / reconcile)."""

import json

import client
import streamlit as st

st.set_page_config(page_title="Ingest", page_icon="📦", layout="wide")
st.title("📦 Ingest & manage")

tab_single, tab_bulk, tab_manage = st.tabs(["Single product", "Bulk upload", "Price / lifecycle"])

# ---------------- single ----------------
with tab_single:
    with st.form("single"):
        c1, c2 = st.columns(2)
        external_id = c1.text_input("external_id *")
        name = c2.text_input("name *")
        brand = c1.text_input("brand")
        category = c2.text_input("category")
        article = c1.text_input("article (SKU)")
        product_code = c2.text_input("product_code")
        ean13 = c1.text_input("ean13")
        currency = c2.text_input("currency", value="UAH")
        price = c1.number_input("price", min_value=0.0, value=0.0, step=1.0)
        in_stock = c2.checkbox("in_stock", value=True)
        description = st.text_area("description")
        attributes_raw = st.text_area(
            "attributes (JSON object)", value='{\n  "Сокет": "LGA1700",\n  "Wi-Fi": false\n}'
        )
        submitted = st.form_submit_button("Upsert product", type="primary")

    if submitted:
        try:
            attributes = json.loads(attributes_raw or "{}")
            if not isinstance(attributes, dict):
                raise ValueError("attributes must be a JSON object")
            product = {
                "external_id": external_id,
                "name": name,
                "description": description or None,
                "brand": brand or None,
                "category": category or None,
                "article": article or None,
                "product_code": product_code or None,
                "ean13": ean13 or None,
                "attributes": attributes,
                "price": price or None,
                "currency": currency or "UAH",
                "in_stock": in_stock,
            }
            res = client.upsert(product)
            st.success(f"Upserted: {res['succeeded']} ok, {res['failed']} failed")
            st.json(res)
        except Exception as exc:
            st.error(str(exc))

# ---------------- bulk ----------------
with tab_bulk:
    st.caption("Paste a JSON array of products, or upload a .json file. Up to 1000 per batch.")
    up = st.file_uploader("JSON file", type=["json"])
    raw = st.text_area("...or paste JSON array here", height=200)
    if st.button("Upsert batch", type="primary"):
        try:
            payload = json.loads(up.read().decode("utf-8")) if up else json.loads(raw or "[]")
            if not isinstance(payload, list):
                raise ValueError("expected a JSON array of products")
            res = client.upsert_batch(payload)
            st.success(f"Batch: {res['succeeded']} ok, {res['failed']} failed of {res['total']}")
            failed = [i for i in res["items"] if not i["ok"]]
            if failed:
                st.warning("Failed items:")
                st.json(failed)
        except Exception as exc:
            st.error(str(exc))

# ---------------- manage ----------------
with tab_manage:
    st.subheader("Price / availability")
    with st.form("price"):
        pid = st.text_input("external_id", key="price_id")
        c1, c2, c3 = st.columns(3)
        new_price = c1.number_input("price", min_value=0.0, value=0.0, step=1.0)
        set_price = c1.checkbox("set price", value=True)
        new_stock = c2.selectbox("in_stock", ["(unchanged)", "true", "false"])
        if st.form_submit_button("Update price / stock"):
            payload: dict = {"external_id": pid}
            if set_price:
                payload["price"] = new_price
            if new_stock != "(unchanged)":
                payload["in_stock"] = new_stock == "true"
            try:
                st.json(client.update_price(pid, payload))
            except Exception as exc:
                st.error(str(exc))

    st.divider()
    st.subheader("Lifecycle")
    ids_raw = st.text_input("external_ids (comma-separated)", key="lc_ids")
    ids = [x.strip() for x in ids_raw.split(",") if x.strip()]
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Archive") and ids:
        st.json(client.archive(ids, True))
    if c2.button("Restore") and ids:
        st.json(client.archive(ids, False))
    if c3.button("Delete", type="secondary") and ids:
        st.json(client.delete(ids))

    st.divider()
    st.subheader("Reconcile (snapshot → archive orphans)")
    snap_raw = st.text_area("valid external_ids (comma or newline separated)", height=100)
    snap = [x.strip() for x in snap_raw.replace("\n", ",").split(",") if x.strip()]
    dry = st.checkbox("dry_run (report only)", value=True)
    cap = st.number_input("max_archived (0 = no cap)", min_value=0, value=0, step=10)
    if st.button("Run reconcile") and snap:
        try:
            st.json(client.reconcile(snap, dry, cap or None))
        except Exception as exc:
            st.error(str(exc))
