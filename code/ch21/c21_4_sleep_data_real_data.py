# Theory of Statistics — Chapter 21: Real Data: Student's Sleep Data

import itertools
import numpy as np
from scipy import stats

# Cushny and Peebles (1905), Student (1908): extra sleep (hours) produced by two drugs in the same 10 patients.
# Source: R, datasets::sleep.
drug1 = np.array([
    0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0,
])
drug2 = np.array([
    1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4,
])
d = np.round(drug2 - drug1, 1)
n = len(d)
print(f"differences d = {d} | mean = {d.mean():.2f}, s = {d.std(ddof=1):.3f}, median = {np.median(d):.2f}")

# 1) Paired t test and 95% confidence interval (Chapter 16)
t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
q = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
print(f"paired t: t = {t:.3f}, p = {2 * stats.t.sf(abs(t), n - 1):.4f}, 95% CI = [{d.mean() - q:.2f}, {d.mean() + q:.2f}]")

# 2) Sign test (the zero difference is dropped)
s = d[d != 0]
k = int(np.sum(s > 0))
print(f"sign test: {k}/{len(s)} positive, p = {stats.binomtest(k, len(s)).pvalue:.4f}")

# 3) Wilcoxon signed-rank test: midranks for ties; exact conditional distribution by enumeration
r = stats.rankdata(np.abs(s))
W = r[s > 0].sum()
signs = np.array(list(itertools.product([0, 1], repeat=len(s))))
W_null = signs @ r
p_w = np.mean(np.abs(W_null - r.sum() / 2) >= abs(W - r.sum() / 2) - 1e-9)
print(f"Wilcoxon: W+ = {W:.1f} (largest possible value {r.sum():.0f}), exact p = {p_w:.4f}")

# 4) Permutation (sign-flip) test: under H0 each difference is equally likely to have either sign
S = np.array(list(itertools.product([-1, 1], repeat=n)))
means = (S * d).mean(axis=1)
print(f"permutation: 2^{n} = {len(means)} sign assignments, p = {np.mean(np.abs(means) >= abs(d.mean()) - 1e-12):.4f}")

# 5) Wrong analysis: a two-sample t test that ignores the pairing
two = stats.ttest_ind(drug2, drug1)
print(f"two-sample t (wrong): t = {two.statistic:.3f}, p = {two.pvalue:.4f} | Corr(drug1, drug2) = {np.corrcoef(drug1, drug2)[0, 1]:.3f}")
