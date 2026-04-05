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

        # Handle potential overflow - cap at a reasonable maximum
        new_equity = equity[-1] * (1 + ret)
        if np.isnan(new_equity) or np.isinf(new_equity) or new_equity > 1e10:
            new_equity = equity[-1]  # Keep previous value if overflow occurs

        equity.append(new_equity)

    return equity[-250:]
