# Theory of Statistics — Chapter 14: Uniform: MLR and the Power of the UMP Test

import numpy as np

rng = np.random.default_rng(42)
theta0, n, a, R = 1.0, 8, 0.05, 200_000
c = theta0 * (1 - a)**(1/n)
for th in [0.9, 1.0, 1.1, 1.3]:
    M = rng.uniform(0, th, size=(R, n)).max(axis=1)
    theory = max(0.0, 1 - (1 - a)*(theta0/th)**n) if th > theta0 else (1 - (c/th)**n if c < th else 0.0)
    print(f"theta={th:.1f} | rejection rate={np.mean(M > c):.4f} | theory={theory:.4f}")
