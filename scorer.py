import numpy as np

def score_etfs(dist, regime, etfs):
    scores = {}
    samples_all = {}

    for etf in etfs:
        vals = dist[regime][etf]

        if len(vals) < 10:
            scores[etf] = -999
            # Generate synthetic samples for edge case - use a small range around 0
            samples_all[etf] = np.random.uniform(-0.05, 0.05, size=100)
            continue

        samples = np.random.choice(vals, size=100, replace=True)
        mu = samples.mean()
        p_up = (samples > 0).mean()

        score = mu * p_up

        scores[etf] = score
        samples_all[etf] = samples

    return scores, samples_all
