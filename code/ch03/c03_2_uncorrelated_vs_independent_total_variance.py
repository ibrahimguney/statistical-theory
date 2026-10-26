# Theory of Statistics — Chapter 3: Uncorrelatedness versus Independence; the Law of Total Variance

import numpy as np

rng = np.random.default_rng(42)
X = rng.normal(size=1_000_000)
Y = X**2
print(f"Corr(X, X^2) = {np.corrcoef(X, Y)[0,1]:+.4f}   (uncorrelated)")
print(f"P(Y>1 | |X|>1) = {np.mean(Y[np.abs(X) > 1] > 1):.1f}  vs  P(Y>1) = {np.mean(Y > 1):.4f}  (dependent)")

# Poisson-Gamma mixture: Var N = E[Var(N|L)] + Var(E[N|L])
alpha, beta = 3.0, 0.5
L = rng.gamma(alpha, 1/beta, size=1_000_000)   # numpy takes scale = 1/rate
N = rng.poisson(L)
print(f"E N   : MC={N.mean():.3f}  theory={alpha/beta:.3f}")
print(f"Var N : MC={N.var():.3f}  theory={alpha/beta + alpha/beta**2:.3f}  (would be {alpha/beta:.1f} for a Poisson)")
