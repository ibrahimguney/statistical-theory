# Theory of Statistics — Chapter 10: The German Tank Problem

import numpy as np

rng = np.random.default_rng(42)
theta, n = 100.0, 10
samples = rng.uniform(0.0, theta, size=(200_000, n))
umvue = ((n + 1.0) / n) * np.max(samples, axis=1)
print(f"UMVUE variance: {np.var(umvue, ddof=1):.4f} (theory: {theta**2/(n*(n+2)):.4f})")
mm = 2 * samples.mean(axis=1)
print(f"MM (2*Xbar) variance: {np.var(mm, ddof=1):.4f} (theory: {theta**2/(3*n):.4f})")
