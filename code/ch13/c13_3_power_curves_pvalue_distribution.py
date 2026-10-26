# Theory of Statistics — Chapter 13: Power Curves and the Distribution of p-Values

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
sigma, za = 2.0, stats.norm.ppf(0.95)
mu = np.linspace(-0.5, 2.5, 300)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for n in [4, 16, 64]:
    ax[0].plot(mu, 1 - stats.norm.cdf(za - np.sqrt(n)*mu/sigma), label=f"n={n}")
ax[0].axhline(0.05, color="gray", ls=":"); ax[0].set_xlabel(r"$\mu$"); ax[0].set_ylabel("power")
ax[0].set_title(r"$H_0:\mu=0$, $\alpha=0.05$"); ax[0].legend()

for mu1, col in [(0.0, "tab:blue"), (0.5, "tab:red")]:
    xb = rng.normal(mu1, sigma/4, size=100_000)
    p = 1 - stats.norm.cdf(xb / (sigma/4))
    ax[1].hist(p, bins=40, range=(0, 1), density=True, alpha=0.6, color=col, label=rf"$\mu={mu1}$")
    print(f"mu={mu1}: P(p <= 0.05) = {np.mean(p <= 0.05):.4f}")
ax[1].set_xlabel("p-value"); ax[1].set_title("distribution of p-values (n=16)"); ax[1].legend()
fig.tight_layout(); fig.savefig("guc_pdeger.pdf")
