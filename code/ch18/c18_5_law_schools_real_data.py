# Theory of Statistics — Chapter 18: Real Data: Law Schools and the Bootstrap of a Correlation

import numpy as np
from scipy import stats

# Efron and Tibshirani (1993): average LSAT score and undergraduate GPA of 82 US law schools
# (the population), and a sample of 15 schools drawn from it. Source: R package bootstrap.
lsat15 = np.array([
    576, 635, 558, 578, 666, 580, 555, 661, 651, 605, 653, 575, 545, 572, 594,
])
gpa15 = np.array([
    3.39, 3.3, 2.81, 3.03, 3.44, 3.07, 3.0, 3.43, 3.36, 3.13, 3.12, 2.74, 2.76, 2.88, 2.96,
])
lsat82 = np.array([
    622, 542, 579, 653, 606, 576, 620, 615, 553, 607, 558, 596, 635, 581, 661, 547, 599, 646,
    622, 611, 546, 614, 628, 575, 662, 627, 608, 632, 587, 581, 605, 704, 477, 591, 578, 572,
    615, 606, 603, 535, 595, 575, 573, 644, 545, 645, 651, 562, 609, 555, 586, 580, 594, 594,
    560, 641, 512, 631, 597, 621, 617, 637, 572, 610, 562, 635, 614, 546, 598, 666, 570, 570,
    605, 565, 686, 608, 595, 590, 558, 611, 564, 575,
])
gpa82 = np.array([
    3.23, 2.83, 3.24, 3.12, 3.09, 3.39, 3.1, 3.4, 2.97, 2.91, 3.11, 3.24, 3.3, 3.22, 3.43, 2.91,
    3.23, 3.47, 3.15, 3.33, 2.99, 3.19, 3.03, 3.01, 3.39, 3.41, 3.04, 3.29, 3.16, 3.17, 3.13,
    3.36, 2.57, 3.02, 3.03, 2.88, 3.37, 3.2, 3.23, 2.98, 3.11, 2.92, 2.85, 3.38, 2.76, 3.27,
    3.36, 3.19, 3.17, 3.0, 3.11, 3.07, 2.96, 3.05, 2.93, 3.28, 3.01, 3.21, 3.32, 3.24, 3.03,
    3.33, 3.08, 3.13, 3.01, 3.3, 3.15, 2.82, 3.2, 3.44, 3.01, 2.92, 3.45, 3.15, 3.5, 3.16, 3.19,
    3.15, 2.81, 3.16, 3.02, 2.74,
])

def corr(a, b):                                             # row-wise correlation
    a = a - a.mean(axis=-1, keepdims=True); b = b - b.mean(axis=-1, keepdims=True)
    return (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1))

rng = np.random.default_rng(42)
n, B = 15, 10_000
r = corr(lsat15, gpa15)
idx = rng.integers(0, n, size=(B, n))
r_b = corr(lsat15[idx], gpa15[idx])
print(f"sample r = {r:.4f} | bootstrap SE = {r_b.std(ddof=1):.4f}, bias = {r_b.mean() - r:+.4f}")
ql, qu = np.percentile(r_b, [2.5, 97.5])
z0 = stats.norm.ppf(np.mean(r_b < r))
jack = np.array([corr(np.delete(lsat15, i), np.delete(gpa15, i)) for i in range(n)])
dj = jack.mean() - jack
a = np.sum(dj**3) / (6 * np.sum(dj**2)**1.5)
zl, zu = stats.norm.ppf([0.025, 0.975])
a1 = stats.norm.cdf(z0 + (z0 + zl) / (1 - a * (z0 + zl)))
a2 = stats.norm.cdf(z0 + (z0 + zu) / (1 - a * (z0 + zu)))
z = np.arctanh(r)
print(f"95% intervals: percentile [{ql:.3f}, {qu:.3f}] | basic [{2*r - qu:.3f}, {2*r - ql:.3f}]"
      f" | BCa [{np.percentile(r_b, 100*a1):.3f}, {np.percentile(r_b, 100*a2):.3f}]"
      f" | Fisher z [{np.tanh(z - 1.96/np.sqrt(n - 3)):.3f}, {np.tanh(z + 1.96/np.sqrt(n - 3)):.3f}]")
print(f"BCa constants: z0 = {z0:.3f}, acceleration a = {a:.3f}")

# True sampling distribution: drawing 15 schools from the population without replacement
R = 100_000
pick = np.argsort(rng.random((R, 82)), axis=1)[:, :n]
r_g = corr(lsat82[pick], gpa82[pick])
print(f"population rho = {corr(lsat82, gpa82):.4f} | true SD(r) = {r_g.std():.4f} | E(r) = {r_g.mean():.4f}")
