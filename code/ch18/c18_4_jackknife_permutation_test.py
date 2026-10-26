# Theory of Statistics — Chapter 18: Jackknife and Permutation Test

import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
x = rng.normal(5, 2, 15)
n = x.size
plug = x.var()                                         # sigma_hat^2 (1/n)
jack = np.array([np.delete(x, i).var() for i in range(n)])
jk = n*plug - (n - 1)*jack.mean()
print(f"sigma_hat^2 = {plug:.5f} | jackknife-corrected = {jk:.5f} | S^2 = {x.var(ddof=1):.5f}")

# A known failure of the jackknife: the SE of the median
med_j = np.array([np.median(np.delete(x, i)) for i in range(n)])
print(f"Number of distinct median jackknife values: {np.unique(med_j).size} (n={n}) -> SE estimate unreliable")

# Permutation test: two skewed samples, with a real difference
def perm_test(a, b, B=9999):
    z = np.r_[a, b]; T0 = b.mean() - a.mean()
    T = np.array([np.diff([p[:a.size].mean(), p[a.size:].mean()])[0]
                  for p in (rng.permutation(z) for _ in range(B))])
    return (1 + np.sum(np.abs(T) >= abs(T0)))/(B + 1)
a = rng.lognormal(0.0, 1.0, 12); b = rng.lognormal(0.8, 1.0, 12)
print(f"Permutation p = {perm_test(a, b):.4f} | Welch t-test p = {stats.ttest_ind(a, b, equal_var=False).pvalue:.4f}")

# Type I error under H0 (small, skewed samples)
rej_p = rej_t = 0; K = 1000
for _ in range(K):
    a = rng.lognormal(0, 1, 8); b = rng.lognormal(0, 1, 8)
    rej_p += perm_test(a, b, B=499) <= 0.05
    rej_t += stats.ttest_ind(a, b, equal_var=False).pvalue <= 0.05
print(f"rejection rate under H0 (target 0.05): permutation = {rej_p/K:.3f} | Welch t = {rej_t/K:.3f}")
