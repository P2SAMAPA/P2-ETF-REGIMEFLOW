"""
Configuration for RegimeFlow ETF Engine.
"""

# Hugging Face dataset
HF_DATASET = "P2SAMAPA/p2-etf-regimeflow-results"
HF_TOKEN = None  # set via environment variable or secrets

# Universes
FI_ETFS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQ_ETFS = ["QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "GDX", "XME", "XBI", "XSD", "IWF"]
COMBINED_ETFS = FI_ETFS + EQ_ETFS

# Benchmark for each universe
FI_BENCHMARK = "AGG"
EQ_BENCHMARK = "SPY"

# Macro features
MACRO_VARS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]

# Regime clustering
N_REGIMES = 4
RANDOM_STATE = 42

# Walk‑forward backtest
TRAIN_SIZE = 1000
TEST_SIZE = 1
CASH_THRESHOLD = 0.01  # 1% expected return threshold to go CASH

# Transaction cost
TRANSACTION_COST = 0.0012  # 12 bps

# Bootstrap parameters
N_BOOTSTRAP = 100
