# Theory of Statistics — Chapter 6: Convergence to the CLT: Skewness and Kurtosis

import numpy as np
import scipy.stats as stats

rng = np.random.default_rng(42)
lam = 0.5
for n in [1, 5, 30, 200]:
    samples = rng.exponential(scale=1.0/lam, size=(100_000, n))
    z = (np.mean(samples, axis=1) - 2.0) / (2.0 / np.sqrt(n))
    print(f"n = {n:3d} | Skewness: {stats.skew(z):6.3f} (theory {2/np.sqrt(n):5.3f}) | Kurtosis: {stats.kurtosis(z):6.3f} (theory {6/n:5.3f})")
