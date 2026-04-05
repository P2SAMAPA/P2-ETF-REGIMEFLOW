import json
import os
from datetime import datetime

from huggingface_hub import HfApi

from config import *
from data_loader import load_data
from regime import compute_regimes
from conditional import build_distributions
from scorer import score_etfs
from backtest import run_backtest
from calendar_utils import get_next_trading_day


df = load_data()
df, model = compute_regimes(df)

# FI
dist_fi = build_distributions(df, FI_ETFS)
regime_now = df.iloc[-1]["regime"]

scores_fi, samples_fi = score_etfs(dist_fi, regime_now, FI_ETFS)
pick_fi = max(scores_fi, key=scores_fi.get)

# EQ
dist_eq = build_distributions(df, EQ_ETFS)
scores_eq, samples_eq = score_etfs(dist_eq, regime_now, EQ_ETFS)
pick_eq = max(scores_eq, key=scores_eq.get)

equity_curve = run_backtest(df, EQ_ETFS, dist_eq)

output = {
    "date": datetime.utcnow().strftime("%Y-%m-%d"),
    "next_trading_day": get_next_trading_day(),

    "FI": {
        "pick": pick_fi,
        "scores": scores_fi
    },

    "EQ": {
        "pick": pick_eq,
        "scores": scores_eq
    },

    "samples_fi": {k: v.tolist() for k, v in samples_fi.items()},
    "samples_eq": {k: v.tolist() for k, v in samples_eq.items()},

    "equity_curve": equity_curve
}

os.makedirs("outputs", exist_ok=True)
fname = f"outputs/regimeflow_{output['date']}.json"

with open(fname, "w") as f:
    json.dump(output, f)

api = HfApi(token=os.environ.get("HF_TOKEN"))

api.upload_file(
    path_or_fileobj=fname,
    path_in_repo=os.path.basename(fname),
    repo_id=HF_OUTPUT_DATASET,
    repo_type="dataset"
)
