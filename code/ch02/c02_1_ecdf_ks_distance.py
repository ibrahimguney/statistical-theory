# Theory of Statistics — Chapter 2: Empirical CDF and the Kolmogorov--Smirnov Distance

import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(42)
x_grid = np.linspace(-3.5, 3.5, 1000)
f_theo = norm.cdf(x_grid)

for n in [20, 100, 1000, 10000]:
    samples = rng.normal(0.0, 1.0, size=n)
    ecdf = np.array([np.mean(samples <= x) for x in x_grid])
    ks_dist = np.max(np.abs(ecdf - f_theo))
    print(f"n = {n:5d} | KS distance: {ks_dist:.4f}")
