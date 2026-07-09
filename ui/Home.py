"""Qdrant Product Search — admin / testing / eval console (Streamlit).

Run: streamlit run ui/Home.py   (env: API_BASE_URL, API_KEY)
"""

import client
import streamlit as st

st.set_page_config(page_title="Product Search Console", page_icon="🔎", layout="wide")

st.title("🔎 Product Search Console")
st.caption(f"API: {client.BASE_URL}")

code, ready = client.ready()
if code == 200:
    st.success(f"Service ready — {ready.get('indexed_code_points', '?')} code-indexed products")
elif code == 0:
    st.error(f"API unreachable: {ready.get('error')}")
else:
    st.warning(f"/ready → {code}: {ready}")

st.subheader("Catalog")
try:
    s = client.stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total", s["total"])
    c2.metric("Active", s["active"])
    c3.metric("Archived", s["archived"])
except Exception as exc:
    st.error(str(exc))

st.divider()
st.markdown(
    "- **Ingest** — add products (form or bulk), update price/availability, archive, reconcile\n"
    "- **Search** — query playground with mode / filters and full match explanation\n"
    "- **Evaluate** — score search quality (nDCG / Recall / MRR per segment) and diff vs the baseline\n\n"
    "Use the sidebar to navigate."
)
