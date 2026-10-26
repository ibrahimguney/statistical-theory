# Theory of Statistics — Chapter 18: Bootstrap Standard Error of the Median

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
n, B = 50, 4000
x = rng.exponential(1.0, size=n)
idx = rng.integers(0, n, size=(B, n))
med_star = np.median(x[idx], axis=1)
mean_star = x[idx].mean(axis=1)

true_se = np.median(rng.exponential(1.0, size=(200_000, n)), axis=1).std()
print(f"Observed median = {np.median(x):.4f} (true ln2 = {np.log(2):.4f})")
print(f"Median SE: bootstrap = {med_star.std(ddof=1):.4f} | true (MC) = {true_se:.4f} | asymptotic 1/sqrt(n) = {1/np.sqrt(n):.4f}")
print(f"Mean SE: bootstrap = {mean_star.std(ddof=1):.4f} | plug-in formula sigma_hat/sqrt(n) = {x.std()/np.sqrt(n):.4f}")
print(f"Proportion of distinct observations in a bootstrap sample ≈ {np.mean([np.unique(r).size for r in idx[:500]])/n:.3f} (1-1/e = {1-np.exp(-1):.3f})")
print(f"Number of distinct values in the bootstrap distribution of the median: {np.unique(med_star).size} / {B}")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ax[0].hist(mean_star, bins=60, color="tab:blue", alpha=0.7); ax[0].set_title(r"Bootstrap distribution of the mean")
ax[1].hist(med_star, bins=60, color="tab:orange", alpha=0.7); ax[1].set_title(r"Bootstrap distribution of the median (discrete)")
for a, s in zip(ax, [mean_star, med_star]):
    for q in np.percentile(s, [2.5, 97.5]):
        a.axvline(q, color="k", ls=":")
fig.tight_layout(); fig.savefig("bootstrap_dagilim.pdf")
