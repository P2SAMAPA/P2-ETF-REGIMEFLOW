import numpy as np

def run_backtest(df, etfs, dist):
    equity = [1.0]

    for i in range(100, len(df)-1):
        regime = df.iloc[i]["regime"]

        scores = {}
        for etf in etfs:
            vals = dist[regime][etf]
            scores[etf] = np.mean(vals)

        pick = max(scores, key=scores.get)
        ret = df.iloc[i+1][pick]

        equity.append(equity[-1]*(1+ret))

    return equity[-250:]
