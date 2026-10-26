# Theory of Statistics — Chapter 15: Verifying Wilks' Theorem (Poisson)

import numpy as np
import scipy.stats as stats

rng = np.random.default_rng(42)
samples = rng.poisson(lam=4.0, size=(150_000, 100))
xb = np.mean(samples, axis=1)
g2 = 2.0 * 100 * (xb * np.log(xb / 4.0) - (xb - 4.0))
print(f"E[G^2]: {np.mean(g2):.4f} (theory: 1.0) | type I error: {np.mean(g2 > stats.chi2.ppf(0.95, 1)):.4f}")
print(f"Var[G^2]: {np.var(g2):.4f} (theory: 2.0) | KS p-value (against chi2_1): {stats.kstest(g2, 'chi2', args=(1,)).pvalue:.3f}")
