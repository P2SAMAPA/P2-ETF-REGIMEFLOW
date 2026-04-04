# app.py

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download

st.set_page_config(layout="wide")

HF_REPO = "P2SAMAPA/p2-etf-regimeflow-results"


# ─────────────────────────────────────────────
# LOAD LATEST
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_latest():
    api = HfApi()

    files = sorted([
        f for f in api.list_repo_files(HF_REPO, repo_type="dataset")
        if f.endswith(".json")
    ])

    if not files:
        return None

    latest = files[-1]

    path = hf_hub_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        filename=latest
    )

    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# LOAD HISTORY
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_history():
    api = HfApi()

    files = sorted([
        f for f in api.list_repo_files(HF_REPO, repo_type="dataset")
        if f.endswith(".json")
    ])

    rows = []

    for f in files[-30:]:
        path = hf_hub_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            filename=f
        )

        with open(path) as file:
            d = json.load(file)

            rows.append({
                "Date": d["date"],
                "FI Pick": d["FI"]["pick"],
                "EQ Pick": d["EQ"]["pick"]
            })

    return pd.DataFrame(rows[::-1])


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
data = load_latest()

if data is None:
    st.warning("No signals yet. Run pipeline first.")
    st.stop()

st.title("REGIMEFLOW — Regime Rotation Engine")
st.caption("Macro Regime Conditioning · Cross-Sectional ETF Selection")


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div style="padding:25px;border-radius:15px;background:#eef4ff;">
        <h2>Fixed Income / Commodities</h2>
        <h1>{data["FI"]["pick"]}</h1>
        <p>Signal for <b>{data["next_trading_day"]}</b></p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="padding:25px;border-radius:15px;background:#fff3e6;">
        <h2>Equity</h2>
        <h1>{data["EQ"]["pick"]}</h1>
        <p>Signal for <b>{data["next_trading_day"]}</b></p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SCORES TABLE
# ─────────────────────────────────────────────
st.markdown("### ETF Scores")

col1, col2 = st.columns(2)

with col1:
    df_fi = pd.DataFrame.from_dict(data["FI"]["scores"], orient="index", columns=["Score"])
    st.dataframe(df_fi.sort_values("Score", ascending=False), use_container_width=True)

with col2:
    df_eq = pd.DataFrame.from_dict(data["EQ"]["scores"], orient="index", columns=["Score"])
    st.dataframe(df_eq.sort_values("Score", ascending=False), use_container_width=True)


# ─────────────────────────────────────────────
# DISTRIBUTION (TOP ETFs)
# ─────────────────────────────────────────────
st.markdown("### Return Distributions (Top FI ETFs)")

top_fi = sorted(data["FI"]["scores"], key=data["FI"]["scores"].get, reverse=True)[:3]

cols = st.columns(3)

for i, etf in enumerate(top_fi):
    vals = data.get("samples_fi", {}).get(etf, [])
    if vals:
        df_plot = pd.DataFrame({"returns": vals})
        cols[i].bar_chart(df_plot)


# ─────────────────────────────────────────────
# EQUITY CURVE
# ─────────────────────────────────────────────
st.markdown("### Strategy Equity Curve")

if data.get("equity_curve"):
    df_eq = pd.DataFrame({"equity": data["equity_curve"]})
    st.line_chart(df_eq)


# ─────────────────────────────────────────────
# HISTORY
# ─────────────────────────────────────────────
st.markdown("### Signal History")

hist = load_history()

if not hist.empty:
    st.dataframe(hist, use_container_width=True)


# ─────────────────────────────────────────────
# REFRESH
# ─────────────────────────────────────────────
if st.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()


st.markdown("---")
st.caption("REGIMEFLOW Engine · Not Financial Advice")
