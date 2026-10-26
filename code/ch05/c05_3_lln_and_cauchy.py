# Theory of Statistics — Chapter 5: The Law of Large Numbers and the Cauchy Counterexample

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
n = np.arange(1, 10_001)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
for r in range(5):
    ax[0].plot(n, np.cumsum(rng.normal(size=n.size)) / n, lw=0.8)
    ax[1].plot(n, np.cumsum(rng.standard_cauchy(size=n.size)) / n, lw=0.8)
ax[0].set_title("Normal(0,1): the mean converges to 0"); ax[1].set_title("Cauchy: the mean does not converge")
for a in ax:
    a.set_xscale("log"); a.axhline(0, color="k", lw=0.6); a.set_xlabel("n")
ax[0].set_ylim(-1.5, 1.5); ax[0].set_ylabel(r"$\bar X_n$")
fig.tight_layout(); fig.savefig("lln_cauchy.pdf")

# The distribution of the Cauchy sample mean does not depend on n:
for m in [1, 10, 1000]:
    xbar = rng.standard_cauchy(size=(20_000, m)).mean(axis=1)
    q = np.percentile(xbar, [25, 75])
    print(f"n={m:5d} | interquartile width of the Cauchy mean: {q[1]-q[0]:.3f} (theory 2.000)")
