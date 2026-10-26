# Theory of Statistics — Chapter 21: Level and Power of the t, Wilcoxon Signed-Rank and Sign Tests

import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
n, M, alpha = 30, 4000, 0.05
dists = {                                    # all with mean 0, variance 1
    "Normal ": lambda size: rng.standard_normal(size),
    "Laplace": lambda size: rng.laplace(0, 1/np.sqrt(2), size),
    "t_3    ": lambda size: rng.standard_t(3, size) / np.sqrt(3),
    "Uniform": lambda size: rng.uniform(-np.sqrt(3), np.sqrt(3), size),
}

def rejection_rates(X):
    t = stats.ttest_1samp(X, 0, axis=1).pvalue < alpha
    w = stats.wilcoxon(X, axis=1).pvalue < alpha
    k = (X > 0).sum(axis=1)
    s = stats.binom.cdf(np.minimum(k, n - k), n, 0.5) * 2 < alpha      # two-sided sign test
    return t.mean(), w.mean(), s.mean()

print("Distrib.| theta | t-test  | Wilcoxon | sign")
for name, gen in dists.items():
    for theta in (0.0, 0.5):
        r = rejection_rates(gen((M, n)) + theta)
        print(f"{name} |  {theta:.1f}  |  {r[0]:.3f}  |  {r[1]:.3f}   | {r[2]:.3f}")

are = {"Normal ": (3/np.pi, 2/np.pi), "Laplace": (1.5, 2.0), "Uniform": (1.0, 1/3)}
print("Theoretical Pitman ARE (Wilcoxon/t, sign/t):",
      {k.strip(): (round(a, 3), round(b, 3)) for k, (a, b) in are.items()})
