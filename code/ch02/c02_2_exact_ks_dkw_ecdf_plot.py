# Theory of Statistics — Chapter 2: Exact KS Distance, the DKW Bound and an ECDF Plot

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
def ks_exact(x, cdf):
    x = np.sort(x); n = len(x); F = cdf(x); i = np.arange(1, n + 1)
    return max(np.max(i / n - F), np.max(F - (i - 1) / n))

eps, R = 0.05, 20_000
for n in [100, 400, 1000]:
    D = np.array([ks_exact(rng.normal(size=n), norm.cdf) for _ in range(R)])
    print(f"n={n:4d} | P(D_n > {eps}) MC = {np.mean(D > eps):.4f} | DKW bound = {min(1, 2*np.exp(-2*n*eps**2)):.4f}"
          f" | E[sqrt(n) D_n] = {np.sqrt(n)*D.mean():.3f}")

fig, ax = plt.subplots(figsize=(7, 3.6))
xg = np.linspace(-3.5, 3.5, 500)
ax.plot(xg, norm.cdf(xg), "k-", lw=2, label="True F (N(0,1))")
for n, c in [(20, "tab:red"), (200, "tab:blue")]:
    s = np.sort(rng.normal(size=n))
    ax.step(np.r_[-3.5, s, 3.5], np.r_[0, np.arange(1, n + 1) / n, 1], where="post", color=c, label=f"ECDF, n={n}")
ax.set_xlabel("x"); ax.set_ylabel("F(x)"); ax.legend(); fig.tight_layout()
fig.savefig("ecdf.pdf")
