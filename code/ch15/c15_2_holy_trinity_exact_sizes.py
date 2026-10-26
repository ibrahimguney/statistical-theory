# Theory of Statistics — Chapter 15: The Holy Trinity: Exact Sizes in Small Samples

import numpy as np
from scipy import stats

def sizes(n, lam0, alpha=0.05):
    t = np.arange(0, int(n*lam0 + 20*np.sqrt(n*lam0) + 30))
    w = stats.poisson.pmf(t, n*lam0)
    xb = t / n
    with np.errstate(divide="ignore", invalid="ignore"):
        G = 2*n*(np.where(xb > 0, xb*np.log(xb/lam0), 0.0) - (xb - lam0))
        W = np.where(xb > 0, n*(xb - lam0)**2 / xb, np.inf)
    S = n*(xb - lam0)**2 / lam0
    k = stats.chi2.ppf(1 - alpha, 1)
    return [np.sum(w[s > k]) for s in (G, W, S)]

print(" n   lambda0 |  LRT    Wald   Score  (target 0.05)")
for n, l0 in [(5, 1.0), (10, 1.0), (10, 0.3), (50, 1.0), (200, 1.0)]:
    g, w, s = sizes(n, l0)
    print(f"{n:4d}  {l0:5.1f}   | {g:.4f} {w:.4f} {s:.4f}")
