# Theory of Statistics — Chapter 2: Inverse Transform and the Probability Integral Transform

import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
U = rng.uniform(size=200_000)
lam = 2.0
X_exp = -np.log(1 - U) / lam                 # Exp(2)
X_cau = np.tan(np.pi * (U - 0.5))            # standard Cauchy
X_poi = np.searchsorted(stats.poisson.cdf(np.arange(60), 3.0), U)  # Poisson(3)

print(f"Exponential: mean={X_exp.mean():.4f} (theory {1/lam}), KS p-value={stats.kstest(X_exp, 'expon', args=(0, 1/lam)).pvalue:.3f}")
print(f"Cauchy     : median={np.median(X_cau):.4f} (theory 0), KS p-value={stats.kstest(X_cau, 'cauchy').pvalue:.3f}")
print(f"Poisson    : mean={X_poi.mean():.4f}, variance={X_poi.var():.4f} (theory 3, 3)")

# Probability integral transform: F(X) ~ U(0,1) (for continuous F)
Z = rng.gamma(3.0, 2.0, size=200_000)      # Gamma(3, rate 1/2); numpy takes scale = 1/rate
V = stats.gamma.cdf(Z, a=3.0, scale=2.0)
print(f"PIT: mean of F(X)={V.mean():.4f} (1/2), variance={V.var():.4f} (1/12={1/12:.4f})")
