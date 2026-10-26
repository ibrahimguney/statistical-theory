# Theory of Statistics — Chapter 14: The Two-Sided Problem: One-Sided Tests Cannot Be UMP

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

sigma, n, a = 1.0, 25, 0.05
d = np.linspace(-1, 1, 401) * np.sqrt(n) / sigma        # sqrt(n)(mu - mu0)/sigma
za, za2 = stats.norm.ppf(1 - a), stats.norm.ppf(1 - a/2)
upp  = 1 - stats.norm.cdf(za - d)
low  = stats.norm.cdf(-za - d)
two  = 1 - stats.norm.cdf(za2 - d) + stats.norm.cdf(-za2 - d)
fig, ax = plt.subplots(figsize=(7, 3.4))
mu = d * sigma / np.sqrt(n)
ax.plot(mu, upp, label="Upper one-sided ($\\bar X > c$)")
ax.plot(mu, low, label="Lower one-sided ($\\bar X < -c$)")
ax.plot(mu, two, "k", lw=2, label="Two-sided (UMPU)")
ax.axhline(a, color="gray", ls=":"); ax.set_xlabel(r"$\mu-\mu_0$"); ax.set_ylabel("power"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("iki_tarafli.pdf")
for m in [-0.4, 0.2, 0.4]:
    k = np.argmin(np.abs(mu - m))
    print(f"mu-mu0={m:+.1f} | upper={upp[k]:.3f} | lower={low[k]:.3f} | two-sided={two[k]:.3f}")
