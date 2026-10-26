# Theory of Statistics — Chapter 17: Verifying the Sampling Theory

import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
n, p, sigma, R = 30, 3, 2.0, 100_000
X = np.c_[np.ones(n), rng.uniform(0, 10, n), rng.normal(5, 2, n)]
beta = np.array([1.0, 0.5, -0.3])
XtX_inv = np.linalg.inv(X.T @ X)
eps = rng.normal(0, sigma, (R, n))
Y = X @ beta + eps
B = Y @ X @ XtX_inv                     # each row is a beta_hat (X'X symmetric)
E = Y - B @ X.T
s2 = np.sum(E**2, axis=1) / (n - p)

print("E[beta_hat]        :", B.mean(0).round(4), "  true:", beta)
print("sd(beta_hat) MC    :", B.std(0).round(4))
print("sd theory          :", (sigma*np.sqrt(np.diag(XtX_inv))).round(4))
print(f"E[s^2] = {s2.mean():.4f} (sigma^2 = {sigma**2}) | Var[(n-p)s^2/sigma^2] = {np.var((n-p)*s2/sigma**2):.3f} (2(n-p) = {2*(n-p)})")
print(f"Corr(beta_hat_1, s^2) = {np.corrcoef(B[:, 1], s2)[0, 1]:+.4f}  (independence)")

tc = stats.t.ppf(0.975, n - p)
se1 = np.sqrt(s2 * XtX_inv[1, 1])
print(f"coverage of 95% t-interval for beta_1 = {np.mean(np.abs(B[:, 1] - beta[1]) <= tc*se1):.4f}")

# F-test: H0: beta_1 = beta_2 = 0 ; size (H0 true) and power (H0 false)
C = np.array([[0, 1, 0], [0, 0, 1]]); q = 2
M = np.linalg.inv(C @ XtX_inv @ C.T)
def F_stat(Bm, s2v):
    D = Bm @ C.T
    return np.einsum("ij,jk,ik->i", D, M, D) / q / s2v
Y0 = X @ np.array([1.0, 0.0, 0.0]) + eps
B0 = Y0 @ X @ XtX_inv; s20 = np.sum((Y0 - B0 @ X.T)**2, 1)/(n - p)
Fk = stats.f.ppf(0.95, q, n - p)
print(f"F-test size = {np.mean(F_stat(B0, s20) > Fk):.4f} (0.05) | power (true beta) = {np.mean(F_stat(B, s2) > Fk):.4f}")
