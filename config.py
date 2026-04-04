# config.py

HF_DATASET = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_OUTPUT_DATASET = "P2SAMAPA/p2-etf-regimeflow-results"

# ── Fixed Income / Commodities ─────────────────────────────
FI_ETFS = [
    "TLT", "LQD", "HYG", "VNQ", "GLD", "SLV"
]
FI_BENCHMARK = "AGG"

# ── Equity ─────────────────────────────────────────────
EQ_ETFS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE",
    "XLV", "XLI", "XLY", "XLP", "XLU",
    "GDX", "XME", "IWM"
]
EQ_BENCHMARK = "SPY"

# ── Macro ─────────────────────────────────────────────
MACRO_VARS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]

# ── Regime Config ─────────────────────────────────────
N_REGIMES = 4
LOOKBACK_REGIME = 252

# ── Portfolio Rules ───────────────────────────────────
TX_COST = 0.0012
TSL_THRESHOLD = -0.12
TSL_WINDOW = 2
CASH_ZSCORE = 0.75
