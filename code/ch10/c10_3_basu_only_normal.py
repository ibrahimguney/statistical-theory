# Theory of Statistics — Chapter 10: Basu's Theorem: bar X indep S^2 Only for the Normal

import numpy as np

rng = np.random.default_rng(42)
n, R = 10, 300_000
dists = {"Normal": lambda: rng.normal(0, 1, (R, n)),
              "Expon. ": lambda: rng.exponential(1, (R, n)),
              "Uniform": lambda: rng.uniform(0, 1, (R, n))}
for name, gen in dists.items():
    X = gen()
    xb, s2 = X.mean(axis=1), X.var(axis=1, ddof=1)
    print(f"{name:7s} | Corr(Xbar, S^2) = {np.corrcoef(xb, s2)[0,1]:+.4f} | Corr(|Xbar-med|, S^2) = "
          f"{np.corrcoef(np.abs(xb - np.median(xb)), s2)[0,1]:+.4f}")
