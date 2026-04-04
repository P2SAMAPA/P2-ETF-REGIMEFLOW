import numpy as np

def build_distributions(df, etfs):
    regimes = df["regime"].unique()

    dist = {}

    for r in regimes:
        dist[r] = {}
        df_r = df[df["regime"] == r]

        for etf in etfs:
            vals = df_r[etf].values
            dist[r][etf] = vals

    return dist
