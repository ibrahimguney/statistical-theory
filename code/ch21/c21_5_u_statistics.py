# Theory of Statistics — Chapter 21: U-Statistics: Exact Variance and Asymptotic Normality

import numpy as np
from scipy import stats, integrate

rng = np.random.default_rng(42)
phi, Phi = stats.norm.pdf, stats.norm.cdf

# 1) Gini's mean difference: h(x1, x2) = |x1 - x2|, data N(0, 1)
theta = 2 / np.sqrt(np.pi)
h1 = lambda x: 2 * phi(x) + x * (2 * Phi(x) - 1)           # h_1(x) = E|x - X|
zeta1 = integrate.quad(lambda x: (h1(x) - theta)**2 * phi(x), -np.inf, np.inf)[0]
zeta2 = 2 - 4 / np.pi                                      # Var|X1 - X2|
print(f"Gini: theta = {theta:.4f} | zeta_1 = {zeta1:.4f}, zeta_2 = {zeta2:.4f} | m^2 zeta_1 = {4*zeta1:.4f}")
R = 20_000
for n in [5, 20, 100]:
    X = np.sort(rng.normal(size=(R, n)), axis=1)
    i = np.arange(1, n + 1)
    U = 2 * np.sum((2 * i - n - 1) * X, axis=1) / (n * (n - 1))  # O(n) computation from sorted data
    exact = 2 / (n * (n - 1)) * (2 * (n - 2) * zeta1 + zeta2)
    print(f"  n={n:3d} | mean(U) = {U.mean():.4f} | n*Var(U) = {n*U.var():.4f}"
          f" | exact n*Var = {n*exact:.4f} | skewness = {stats.skew(U):+.3f}")

# 2) Walsh pairs: h(x1, x2) = I{x1 + x2 > 0}, data N(0.3, 1) (the Wilcoxon U-statistic)
d = 0.3
p = Phi(2 * d / np.sqrt(2))                                # P(X1 + X2 > 0)
zeta1 = integrate.quad(lambda x: (Phi(x + d) - p)**2 * phi(x - d), -np.inf, np.inf)[0]
print(f"Walsh: p = {p:.4f} | 4 zeta_1 = {4*zeta1:.4f} (1/3 = 0.3333 under H0)")
for n in [10, 50]:
    X = rng.normal(d, 1, size=(R, n))
    iu = np.triu_indices(n, 1)
    U = ((X[:, iu[0]] + X[:, iu[1]]) > 0).mean(axis=1)
    exact = 2 / (n * (n - 1)) * (2 * (n - 2) * zeta1 + p * (1 - p))   # zeta_2 = Var h = p(1 - p)
    z = np.sqrt(n) * (U - p) / np.sqrt(4 * zeta1)
    print(f"  n={n:3d} | mean(U) = {U.mean():.4f} | n*Var(U) = {n*U.var():.4f} | exact n*Var = {n*exact:.4f}"
          f" | P(|z| > 1.96) = {np.mean(np.abs(z) > 1.96):.4f}")
