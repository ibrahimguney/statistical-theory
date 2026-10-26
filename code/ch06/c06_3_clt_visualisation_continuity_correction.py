# Theory of Statistics — Chapter 6: Visualising the CLT and the Continuity Correction

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
fig, axes = plt.subplots(1, 4, figsize=(11, 2.8), sharey=True)
zz = np.linspace(-4, 4, 300)
for ax, n in zip(axes, [1, 3, 10, 50]):
    z = (rng.exponential(size=(200_000, n)).mean(axis=1) - 1) * np.sqrt(n)
    ax.hist(z, bins=120, range=(-4, 6), density=True, color="tab:blue", alpha=0.6)
    ax.plot(zz, stats.norm.pdf(zz), "k-", lw=1.5)
    ax.set_title(f"n = {n}"); ax.set_xlim(-4, 6)
fig.tight_layout(); fig.savefig("clt_ustel.pdf")

n, p, k = 100, 0.3, 25
mu, sd = n*p, np.sqrt(n*p*(1-p))
print(f"Exact P(X<=25)                    = {stats.binom.cdf(k, n, p):.4f}")
print(f"Normal approx. (no correction)    = {stats.norm.cdf((k - mu)/sd):.4f}")
print(f"Normal approx. (with correction)  = {stats.norm.cdf((k + 0.5 - mu)/sd):.4f}")
