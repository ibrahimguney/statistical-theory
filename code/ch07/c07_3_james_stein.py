# Theory of Statistics — Chapter 7: Inadmissibility: the James--Stein Estimator

import numpy as np
from scipy import stats

def exact_risk(norm_theta, k):
    # Theorem: R = k - (k-2)^2 E[1/||X||^2]; ||X||^2 is noncentral chi-square,
    # E[1/||X||^2] = E[1/(k-2+2J)], J ~ Poisson(||theta||^2 / 2)
    j = np.arange(400)
    expected = np.sum(stats.poisson.pmf(j, norm_theta**2 / 2) / (k - 2 + 2 * j))
    return k - (k - 2)**2 * expected

rng = np.random.default_rng(42)
k, R = 10, 200_000
for norm_theta in [0.0, 1.0, 3.0, 10.0]:
    theta = np.full(k, norm_theta / np.sqrt(k))
    X = theta + rng.normal(size=(R, k))
    js = (1 - (k - 2) / np.sum(X**2, axis=1, keepdims=True)) * X
    r_mle = np.mean(np.sum((X - theta)**2, axis=1))
    r_js = np.mean(np.sum((js - theta)**2, axis=1))
    upper = k - (k - 2)**2 / (k + norm_theta**2)
    print(f"||theta||={norm_theta:5.1f} | Risk(X)={r_mle:6.3f} | Risk(JS)={r_js:6.3f}"
          f" | exact={exact_risk(norm_theta, k):6.3f} | upper bound={upper:6.3f}")
