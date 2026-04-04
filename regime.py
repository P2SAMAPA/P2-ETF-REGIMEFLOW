import numpy as np
from sklearn.cluster import KMeans
from config import MACRO_VARS, N_REGIMES

def compute_regimes(df):
    X = df[MACRO_VARS].values

    kmeans = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=10)
    regimes = kmeans.fit_predict(X)

    df["regime"] = regimes
    return df, kmeans
