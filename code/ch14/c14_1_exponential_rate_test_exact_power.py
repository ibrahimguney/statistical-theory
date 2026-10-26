# Theory of Statistics — Chapter 14: The Exponential Rate Test: Simulation and Exact Power

import numpy as np
import scipy.stats as stats

rng = np.random.default_rng(42)
c = stats.chi2.ppf(0.05, df=20) / 2.0
samples = rng.exponential(scale=1.0/1.5, size=(100_000, 10))
print(f"theta=1.5 power: {np.mean(np.sum(samples, axis=1) < c):.4f}")
print(f"Critical value c = {c:.4f} | exact power = P(chi2_20 < 3c) = {stats.chi2.cdf(1.5*2*c, 20):.4f}")
h0 = rng.exponential(scale=1.0, size=(100_000, 10))
print(f"rejection rate at theta=1.0 (size): {np.mean(h0.sum(axis=1) < c):.4f}")
for th in [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
    print(f"  beta({th:.1f}) = {stats.chi2.cdf(2*th*c, 20):.4f}")
