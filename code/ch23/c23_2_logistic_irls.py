# Theory of Statistics — Chapter 23: IRLS (Fisher Scoring) for Logistic Regression and Coverage of Wald Intervals

import numpy as np
from scipy import optimize, special

rng = np.random.default_rng(42)

def irls(X, y, tol=1e-10, maxit=50):
    beta = np.zeros(X.shape[1])
    for it in range(maxit):
        eta = X @ beta; mu = special.expit(eta); W = mu * (1 - mu)
        z = eta + (y - mu) / W                            # working response
        new = np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (W * z))
        if np.max(np.abs(new - beta)) < tol:
            beta = new; break
        beta = new
    mu = special.expit(X @ beta); W = mu * (1 - mu)
    cov = np.linalg.inv(X.T @ (W[:, None] * X))
    return beta, cov, it + 1

n, beta0 = 200, np.array([-0.5, 1.0, -2.0])
X = np.column_stack([np.ones(n), rng.normal(size=n), rng.uniform(-1, 1, n)])
y = rng.random(n) < special.expit(X @ beta0)
b, V, it = irls(X, y)
neg_ll = lambda bb: -(y * (X @ bb) - np.logaddexp(0, X @ bb)).sum()
b_opt = optimize.minimize(neg_ll, np.zeros(3), method="BFGS", options={"gtol": 1e-8}).x
mu = special.expit(X @ b)
dev = 2 * np.sum(special.xlogy(y, y / mu) + special.xlogy(1 - y, (1 - y) / (1 - mu)))
print(f"IRLS converged in {it} iterations: beta_hat = {np.round(b, 4)}")
print(f"General-purpose optimiser (BFGS):  beta_hat = {np.round(b_opt, 4)}")
print(f"Standard errors sqrt(diag((X'WX)^-1)) = {np.round(np.sqrt(np.diag(V)), 4)}")
print(f"Deviance = {dev:.3f} | degrees of freedom n - p = {n - 3}")

M, covers = 2000, np.zeros(3)
for _ in range(M):
    Xs = np.column_stack([np.ones(n), rng.normal(size=n), rng.uniform(-1, 1, n)])
    ys = rng.random(n) < special.expit(Xs @ beta0)
    bs, Vs, _ = irls(Xs, ys)
    covers += np.abs(bs - beta0) <= 1.96 * np.sqrt(np.diag(Vs))
print(f"Coverage of 95% Wald intervals (M = {M}, n = {n}): {np.round(covers / M, 3)}")

# Complete separation: always 1 for x > 0, always 0 for x < 0 -> no MLE
xa = np.r_[rng.uniform(-2, -0.1, 20), rng.uniform(0.1, 2, 20)]
Xa = np.column_stack([np.ones(40), xa]); ya = (xa > 0).astype(float)
for k in (2, 4, 6, 8):
    bk, _, _ = irls(Xa, ya, maxit=k)
    ll = (ya * (Xa @ bk) - np.logaddexp(0, Xa @ bk)).sum()
    print(f"Complete separation, {k:2d} IRLS steps: slope = {bk[1]:9.2f} | log-likelihood = {ll:.2e}")
