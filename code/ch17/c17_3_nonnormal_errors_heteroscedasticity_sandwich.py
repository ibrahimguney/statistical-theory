# Theory of Statistics — Chapter 17: Non-Normal Errors, Heteroscedasticity and Sandwich Standard Errors

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
R = 20_000
def coverage(n, err, sandwich=False):
    x = np.linspace(0, 10, n); X = np.c_[np.ones(n), x]
    XtX_inv = np.linalg.inv(X.T @ X)
    eps = err(x, (R, n))
    Y = X @ np.array([1.0, 0.5]) + eps
    B = Y @ X @ XtX_inv; E = Y - B @ X.T
    if sandwich:
        meat = np.einsum("ri,ij,ik->rjk", E**2, X, X)            # sum e_i^2 x_i x_i'
        V = XtX_inv @ meat @ XtX_inv
        se = np.sqrt(V[:, 1, 1] * n/(n - 2))                     # HC1 correction
    else:
        se = np.sqrt(np.sum(E**2, 1)/(n - 2) * XtX_inv[1, 1])
    return np.mean(np.abs(B[:, 1] - 0.5) <= stats.t.ppf(0.975, n - 2)*se)

expo = lambda x, sh: rng.exponential(1, sh) - 1                 # skewed, constant variance
hetero = lambda x, sh: rng.normal(0, 1, sh) * (0.2 + 0.3*x)   # variance increases with x
for n in [10, 50, 200]:
    print(f"n={n:3d} | skewed errors: classical={coverage(n, expo):.3f} | heteroscedastic: classical={coverage(n, hetero):.3f},"
          f" sandwich={coverage(n, hetero, True):.3f}")

# Confidence and prediction bands
n = 25; x = np.sort(rng.uniform(0, 10, n)); X = np.c_[np.ones(n), x]
y = 1 + 0.5*x + rng.normal(0, 1, n)
bh = np.linalg.solve(X.T @ X, X.T @ y); s = np.sqrt(np.sum((y - X @ bh)**2)/(n - 2))
xx = np.linspace(-1, 11, 200); X0 = np.c_[np.ones_like(xx), xx]
q0 = np.einsum("ij,jk,ik->i", X0, np.linalg.inv(X.T @ X), X0); t = stats.t.ppf(0.975, n - 2)
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.scatter(x, y, s=14, color="k", zorder=3, label="data")
ax.plot(xx, X0 @ bh, "b", label="LS line")
ax.fill_between(xx, X0 @ bh - t*s*np.sqrt(q0), X0 @ bh + t*s*np.sqrt(q0), alpha=0.35, color="tab:blue", label="95% confidence band (mean)")
ax.fill_between(xx, X0 @ bh - t*s*np.sqrt(1 + q0), X0 @ bh + t*s*np.sqrt(1 + q0), alpha=0.15, color="tab:orange", label="95% prediction band (new observation)")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(fontsize=8, loc="upper left"); fig.tight_layout(); fig.savefig("regresyon_bant.pdf")
