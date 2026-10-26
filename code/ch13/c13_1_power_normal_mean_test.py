# Theory of Statistics — Chapter 13: Power of the Test for a Normal Mean

import numpy as np
import scipy.stats as stats

rng = np.random.default_rng(42)
s_h0 = rng.normal(0.0, 2.0, size=(150_000, 16))
s_h1 = rng.normal(1.0, 2.0, size=(150_000, 16))
c = stats.norm.ppf(0.95) * (2.0 / np.sqrt(16))
print(f"Critical value c = {c:.4f}")
print(f"Type I error (MC): {np.mean(np.mean(s_h0, axis=1) > c):.4f}  (target 0.05)")
print(f"Neyman-Pearson power: {np.mean(np.mean(s_h1, axis=1) > c):.4f}")
print(f"Analytic power       : {1 - stats.norm.cdf(stats.norm.ppf(0.95) - np.sqrt(16)*1.0/2.0):.4f}")
za, zb = stats.norm.ppf(0.95), stats.norm.ppf(0.80)
print(f"Required n for power 0.80 = {((za + zb) * 2.0 / 1.0)**2:.2f}")
