# Theory of Statistics — Chapter 12: Asymptotic Normality of the MLE and Coverage of the Wald Interval

import numpy as np
from scipy.special import digamma, polygamma
from scipy import stats

rng = np.random.default_rng(42)
alpha0, R = 2.0, 50_000
def mle_alpha(lx_bar):
    a = np.full_like(lx_bar, 1.0)
    for _ in range(50):                                   # vectorised Newton
        a = np.maximum(a - (digamma(a) - lx_bar) / polygamma(1, a), 1e-6)
    return a
for n in [10, 50, 400]:
    X = rng.gamma(alpha0, 1.0, size=(R, n))
    a_hat = mle_alpha(np.log(X).mean(axis=1))
    z = np.sqrt(n * polygamma(1, alpha0)) * (a_hat - alpha0)
    se = 1 / np.sqrt(n * polygamma(1, a_hat))
    coverage = np.mean(np.abs(a_hat - alpha0) <= 1.96 * se)
    print(f"n={n:4d} | bias={a_hat.mean()-alpha0:+.4f} | n*Var={n*a_hat.var():.4f} (1/I={1/polygamma(1,alpha0):.4f})"
          f" | skewness(z)={stats.skew(z):+.3f} | Wald 95% coverage={coverage:.4f}")
