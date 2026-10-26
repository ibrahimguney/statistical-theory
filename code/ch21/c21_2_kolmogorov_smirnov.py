# Theory of Statistics — Chapter 21: Distribution-Freeness of the Kolmogorov--Smirnov Statistic and the Lilliefors Problem

import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
n, M = 20, 20_000

def D(x, cdf):                                    # one-sample KS statistic sup|F_n - F|
    x = np.sort(x, axis=1); u = cdf(x); i = np.arange(1, n + 1)
    return np.maximum((i / n - u).max(axis=1), (u - (i - 1) / n).max(axis=1))

D_norm = D(rng.standard_normal((M, n)), stats.norm.cdf)
D_exp = D(rng.exponential(1.0, (M, n)), stats.expon.cdf)
q = [0.5, 0.9, 0.95, 0.99]
print("Quantiles        :", q)
print("D_n, normal data :", np.round(np.quantile(D_norm, q), 4))
print("D_n, expon. data :", np.round(np.quantile(D_exp, q), 4))
print("Exact (kstwo)    :", np.round(stats.kstwo.ppf(q, n), 4))
print("Asymptotic K/sqrt(n):", np.round(stats.kstwobign.ppf(q) / np.sqrt(n), 4))

# Lilliefors: KS critical values become invalid when parameters are estimated from the data
X = rng.normal(5, 2, (M, n))
mu, s = X.mean(axis=1, keepdims=True), X.std(axis=1, ddof=1, keepdims=True)
D_est = D((X - mu) / s, stats.norm.cdf)
crit = stats.kstwo.ppf(0.95, n)
print(f"H0 true, parameters estimated: rejection rate (target 0.05) = {(D_est > crit).mean():.4f}")
print(f"0.95 quantile of the estimated statistic = {np.quantile(D_est, 0.95):.4f} (KS critical value {crit:.4f})")
