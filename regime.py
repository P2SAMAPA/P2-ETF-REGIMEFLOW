import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from config import MACRO_VARS, N_REGIMES, REGIME_LOOKBACK_WINDOW


def compute_regimes(df):
    """
    Compute market regimes using Rolling, Scaled KMeans clustering on macro variables.
    
    v2.0 Fixes:
    1. StandardScaler: Prevents high-magnitude variables (like yields) from drowning out 
       low-magnitude variables (like VIX), which was causing the model to freeze.
    2. Rolling Window: Trains only on the most recent REGIME_LOOKBACK_WINDOW days. This allows 
       the cluster centroids to adapt to the *current* macro regime, rather than being 
       permanently anchored to ancient data.
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

    # -----------------------------------------------------------------
    # CRITICAL FIX 1: Rolling Window
    # -----------------------------------------------------------------
    # If we train on 20 years of data, centroids are anchored to the past.
    # We only train on the recent window so centroids can shift with current regimes.
    if len(df_valid) > REGIME_LOOKBACK_WINDOW:
        df_model = df_valid.iloc[-REGIME_LOOKBACK_WINDOW:]
    else:
        df_model = df_valid

    X = df_model[valid_macro_vars].values

    # -----------------------------------------------------------------
    # CRITICAL FIX 2: Feature Scaling
    # -----------------------------------------------------------------
    # Standardize to mean=0, variance=1 so VIX and Yields have equal voting power.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=10)
    regimes = kmeans.fit_predict(X_scaled)

    df_model["regime"] = regimes

    # Assign regimes back to original dataframe
    df["regime"] = np.nan
    df.loc[df_model.index, "regime"] = df_model["regime"].values

    # Fill missing regimes with the most common regime from the recent window
    if df["regime"].isna().any():
        most_common_regime = int(pd.Series(regimes).mode()[0])
        df["regime"] = df["regime"].fillna(most_common_regime)

    df["regime"] = df["regime"].astype(int)

    # Return signature remains identical for backward compatibility
    return df, kmeans
