# Theory of Statistics — Chapter 16: Coverage of the t Interval

import numpy as np
import scipy.stats as stats

rng = np.random.default_rng(42)
s = rng.normal(50.0, 10.0, size=(100_000, 15))
xb, se = np.mean(s, axis=1), np.std(s, axis=1, ddof=1) / np.sqrt(15)
tc = stats.t.ppf(0.975, df=14)
print(f"Coverage: {np.mean((xb - tc*se <= 50.0) & (50.0 <= xb + tc*se)) * 100:.2f}%")
zc = stats.norm.ppf(0.975)
print(f"with z instead (wrong): {np.mean(np.abs(xb - 50.0) <= zc*se) * 100:.2f}%")

# Departure from normality: the t interval for the mean is robust, the variance interval is not
for name, gen, mu, var in [("Normal", lambda: rng.normal(0, 1, (100_000, 15)), 0, 1),
                          ("Expon.", lambda: rng.exponential(1, (100_000, 15)), 1, 1)]:
    X = gen(); xb = X.mean(1); S2 = X.var(1, ddof=1)
    ct = np.mean(np.abs(xb - mu) <= tc*np.sqrt(S2/15))
    lo, hi = 14*S2/stats.chi2.ppf(0.975, 14), 14*S2/stats.chi2.ppf(0.025, 14)
    cv = np.mean((lo <= var) & (var <= hi))
    print(f"{name} | t-interval coverage={ct:.4f} | chi-square variance interval coverage={cv:.4f}  (target 0.95)")
