# Theory of Statistics — Chapter 10: Checking the Characterisation of the UMVUE

import numpy as np

rng = np.random.default_rng(42)
p, n, R = 0.3, 6, 1_000_000
X = rng.binomial(1, p, size=(R, n))
U = X[:, 0] - X[:, 1]
T = X.sum(axis=1)
print(f"Cov(Xbar, U) = {np.cov(X.mean(axis=1), U)[0,1]:+.5f}  (UMVUE -> 0)")
print(f"Cov(X_1,  U) = {np.cov(X[:, 0], U)[0,1]:+.5f}  (theory p(1-p) = {p*(1-p):.5f})")
est = T*(n - T)/(n*(n - 1))
print(f"T(n-T)/(n(n-1)) mean = {est.mean():.5f} | target p(1-p) = {p*(1-p):.5f}")
print(f"Xbar(1-Xbar)    mean = {(T/n*(1 - T/n)).mean():.5f} | theory {(n-1)/n*p*(1-p):.5f}")
