# Theory of Statistics — Chapter 5: An Empirical Check of Chebyshev's Bound

import numpy as np

rng = np.random.default_rng(42)
samples = rng.gamma(shape=2.0, scale=2.0, size=500_000)   # rate 1/2 <=> scale 2
sigma = np.sqrt(8.0)
for k in [1.5, 2.0, 3.0, 4.0]:
    print(f"k = {k:.1f} | Empirical: {np.mean(np.abs(samples - 4.0) >= k * sigma):.4f} <= Chebyshev: {1.0/(k**2):.4f}")
