# Theory of Statistics — Chapter 1: Verifying Continuity by Monte Carlo

import numpy as np

rng = np.random.default_rng(42)
N_SAMPLES = 500_000
omega_samples = rng.uniform(0.0, 1.0, size=N_SAMPLES)

for n in [1, 5, 10, 25, 50]:
    threshold = 1.0 - (1.0 / n)
    p_sim = np.mean(omega_samples <= threshold)
    print(f"n = {n:2d} | Theoretical P(A_n): {threshold:.4f} | Monte Carlo: {p_sim:.4f}")
