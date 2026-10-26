# Theory of Statistics — Chapter 7: Comparing the MSE of S^2 and the MLE

import numpy as np

rng = np.random.default_rng(42)
n, s2_true = 5, 4.0
samples = rng.normal(10.0, np.sqrt(s2_true), size=(200_000, n))
s2 = np.var(samples, axis=1, ddof=1)
mle = np.var(samples, axis=1, ddof=0)
print(f"S^2 MSE: {np.mean((s2 - s2_true)**2):.4f} | MLE MSE: {np.mean((mle - s2_true)**2):.4f}")

# Family of divisors: SS / d , d = n-1, n, n+1 -- bias, variance, MSE
SS = np.sum((samples - samples.mean(axis=1, keepdims=True))**2, axis=1)
for d in [n - 1, n, n + 1]:
    est = SS / d
    print(f"divisor={d} | bias={est.mean()-s2_true:+.4f} | var={est.var():.4f} | MSE={np.mean((est-s2_true)**2):.4f}"
          f" | theory MSE={((d-n+1)**2 + 2*(n-1)) * s2_true**2 / d**2:.4f}")
