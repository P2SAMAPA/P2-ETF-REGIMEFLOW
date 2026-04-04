import pandas as pd
from datasets import load_dataset
from config import MACRO_VARS

def load_data():
    ds = load_dataset("P2SAMAPA/fi-etf-macro-signal-master-data")
    df = ds["train"].to_pandas()

    if "__index_level_0__" in df.columns:
        df = df.rename(columns={"__index_level_0__": "date"})

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    df = df.dropna()

    return df
