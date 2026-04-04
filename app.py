import streamlit as st
import json
from huggingface_hub import HfApi, hf_hub_download

HF_REPO = "P2SAMAPA/p2-etf-regimeflow-results"

@st.cache_data
def load_latest():
    api = HfApi()
    files = sorted([f for f in api.list_repo_files(HF_REPO, repo_type="dataset") if f.endswith(".json")])
    path = hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename=files[-1])
    with open(path) as f:
        return json.load(f)

data = load_latest()

st.title("REGIMEFLOW ETF Engine")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Fixed Income / Commodities")
    st.metric("Pick", data["FI"]["pick"])

with col2:
    st.subheader("Equity")
    st.metric("Pick", data["EQ"]["pick"])

st.subheader("Equity Curve")
st.line_chart(data["equity_curve"])
