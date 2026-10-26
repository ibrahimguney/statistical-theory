# Theory of Statistics — Chapter 9: Uniform(0,theta): Rao--Blackwell by Simulation

import numpy as np

rng = np.random.default_rng(42)
theta, n, R = 5.0, 8, 300_000
X = rng.uniform(0, theta, size=(R, n))
W = 2 * X[:, 0]
M = X.max(axis=1)
W_star = (n + 1) / n * M
# Check the function E[W | M] empirically: split M into slices
bins = np.quantile(M, np.linspace(0, 1, 6))
idx = np.digitize(M, bins[1:-1])
for k in range(5):
    sel = idx == k
    print(f"M slice {k}: mean(W | M) = {W[sel].mean():.3f}  vs  (n+1)/n * mean(M) = {(n+1)/n*M[sel].mean():.3f}")
print(f"Var(W) = {W.var():.4f} (theory {theta**2/3:.4f}) | Var(W*) = {W_star.var():.4f} (theory {theta**2/(n*(n+2)):.4f})")
