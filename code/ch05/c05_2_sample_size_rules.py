# Theory of Statistics — Chapter 5: How Well Do the Sample-Size Rules Really Perform?

import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
eps, delta, R = 0.05, 0.05, 200_000
rules = {"Chebyshev": int(np.ceil(1/(4*eps**2*delta))),
            "Hoeffding": int(np.ceil(np.log(2/delta)/(2*eps**2))),
            "CLT":       int(np.ceil((stats.norm.ppf(1-delta/2)/(2*eps))**2))}
for p in [0.5, 0.2]:
    for name, n in rules.items():
        p_hat = rng.binomial(n, p, size=R) / n
        err = np.mean(np.abs(p_hat - p) >= eps)
        print(f"p={p} | {name:9s} n={n:4d} | P(|p_hat-p|>=0.05) = {err:.4f} (target <= {delta})")
