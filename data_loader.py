import pandas as pd
from datasets import load_dataset
from config import MACRO_VARS


def load_data():
    """
    Load data from HuggingFace dataset with handling for always-null columns.
    
    Filters out rows where ALL macro variables are NaN, but dynamically
    adapts to columns that are entirely null in the dataset.
    """
    ds = load_dataset("P2SAMAPA/fi-etf-macro-signal-master-data")
    df = ds["train"].to_pandas()

    if "__index_level_0__" in df.columns:
        df = df.rename(columns={"__index_level_0__": "date"})

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Identify which macro vars have actual data (not all-null)
    available_macro_vars = [col for col in MACRO_VARS if col in df.columns]
    valid_macro_vars = [
        col for col in available_macro_vars
        if df[col].notna().sum() > 0
    ]

    all_null_cols = [
        col for col in available_macro_vars
        if df[col].notna().sum() == 0
    ]

    if all_null_cols:
        print(f"[DataLoader] WARNING - All-null columns excluded: {all_null_cols}")

    if len(valid_macro_vars) == 0:
        raise ValueError(
            f"CRITICAL: All macro variables are null in the dataset! "
            f"Checked: {MACRO_VARS}"
        )

    print(f"[DataLoader] Using macro vars: {valid_macro_vars}")

    # Drop rows where ALL valid macro vars are NaN
    df_clean = df.dropna(subset=valid_macro_vars)

    rows_removed = len(df) - len(df_clean)
    print(f"[DataLoader] Total rows: {len(df)}")
    print(f"[DataLoader] Rows removed: {rows_removed}")
    print(f"[DataLoader] Remaining rows: {len(df_clean)}")

    if len(df_clean) == 0:
        raise ValueError(
            f"No valid rows after dropping NaN from {valid_macro_vars}"
        )

    return df_clean
