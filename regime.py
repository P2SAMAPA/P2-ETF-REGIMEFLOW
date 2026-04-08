import numpy as np
from sklearn.cluster import KMeans
from config import MACRO_VARS, N_REGIMES


def compute_regimes(df):
    """
    Compute market regimes using KMeans clustering on macro variables.
    
    Dynamically filters out macro vars that are entirely null to prevent
    KMeans from receiving an empty array.
    """
    # Filter to macro vars that exist and have data
    valid_macro_vars = [
        col for col in MACRO_VARS
        if col in df.columns and df[col].notna().sum() > 0
    ]

    if len(valid_macro_vars) == 0:
        raise ValueError(
            f"No valid macro variables found. "
            f"Available columns: {MACRO_VARS}. "
            f"Check if upstream dataset has data."
        )

    # Filter to rows with valid data in all available macro vars
    valid_mask = df[valid_macro_vars].notna().all(axis=1)
    df_valid = df[valid_mask].copy()

    if len(df_valid) == 0:
        raise ValueError(
            f"No valid rows after filtering for macro vars: {valid_macro_vars}"
        )

    if len(df_valid) < N_REGIMES:
        raise ValueError(
            f"Insufficient data: {len(df_valid)} rows, need at least {N_REGIMES}"
        )

    X = df_valid[valid_macro_vars].values
    kmeans = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=10)
    regimes = kmeans.fit_predict(X)

    df_valid["regime"] = regimes

    # Assign regimes back to original dataframe
    df["regime"] = np.nan
    df.loc[valid_mask, "regime"] = df_valid["regime"].values

    # Fill missing regimes with the most common regime
    if df["regime"].isna().any():
        most_common_regime = int(pd.Series(regimes).mode()[0])
        df["regime"] = df["regime"].fillna(most_common_regime)

    df["regime"] = df["regime"].astype(int)

    return df, kmeans
