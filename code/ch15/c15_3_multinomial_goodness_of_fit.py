# Theory of Statistics — Chapter 15: Multinomial Goodness of Fit

import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
p0 = np.array([0.1, 0.2, 0.3, 0.4])
n, R = 40, 200_000
N = rng.multinomial(n, p0, size=R)
E = n * p0
with np.errstate(divide="ignore", invalid="ignore"):
    G2 = 2 * np.sum(np.where(N > 0, N * np.log(N / E), 0.0), axis=1)
X2 = np.sum((N - E)**2 / E, axis=1)
k = stats.chi2.ppf(0.95, df=3)
print(f"n={n}: P(G^2 > chi2_3,0.05) = {np.mean(G2 > k):.4f} | P(X^2 > chi2_3,0.05) = {np.mean(X2 > k):.4f}")
print(f"means: G^2={G2.mean():.3f}, X^2={X2.mean():.3f} (theory 3) | Corr(G^2, X^2)={np.corrcoef(G2, X2)[0,1]:.4f}")
# An example data set
observed = np.array([7, 5, 13, 15])
r1 = stats.chisquare(observed, E)
r2 = stats.power_divergence(observed, E, lambda_="log-likelihood")
print(f"Observed {observed}, expected {E}")
print(f"Pearson X^2 = {r1.statistic:.4f}, p = {r1.pvalue:.4f} | G^2 = {r2.statistic:.4f}, p = {r2.pvalue:.4f}")
