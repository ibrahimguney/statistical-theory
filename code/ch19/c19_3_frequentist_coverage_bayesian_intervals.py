# Theory of Statistics — Chapter 19: Frequentist Coverage of Bayesian Intervals

import numpy as np
from scipy import stats

def coverage_jeffreys(n, ps, a=0.05):
    x = np.arange(n + 1)
    L = np.where(x > 0, stats.beta.ppf(a/2, x + 0.5, n - x + 0.5), 0.0)
    U = np.where(x < n, stats.beta.ppf(1 - a/2, x + 0.5, n - x + 0.5), 1.0)
    return np.array([np.sum(stats.binom.pmf(x, n, p) * ((L <= p) & (p <= U))) for p in ps])

ps = np.linspace(0.005, 0.995, 991)
for n in [30, 100]:
    k = coverage_jeffreys(n, ps)
    print(f"n={n:3d} | Jeffreys interval: mean coverage={k.mean():.4f}, min={k.min():.4f}")
