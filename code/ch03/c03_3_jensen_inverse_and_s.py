# Theory of Statistics — Chapter 3: Jensen's Inequality: E[1/X] and E[S]

import numpy as np
from scipy.special import gammaln

rng = np.random.default_rng(42)
X = rng.gamma(5.0, 1.0, size=1_000_000)       # E X = 5
print(f"E[1/X] = {np.mean(1/X):.4f} >= 1/E[X] = {1/X.mean():.4f}  (theory E[1/X]=1/4)")

n, sigma = 5, 2.0
S = np.std(rng.normal(0, sigma, size=(500_000, n)), axis=1, ddof=1)
c4 = np.sqrt(2/(n-1)) * np.exp(gammaln(n/2) - gammaln((n-1)/2))
print(f"E[S] MC = {S.mean():.4f} < sigma = {sigma}  | theory c4*sigma = {c4*sigma:.4f}")
