# Theory of Statistics — Chapter 1: The Birthday Problem: Exact Value and Simulation

import numpy as np

rng = np.random.default_rng(42)
def exact(n):
    return 1.0 - np.prod((365 - np.arange(n)) / 365)

R = 100_000
for n in [10, 23, 30, 50]:
    days = rng.integers(0, 365, size=(R, n))
    days.sort(axis=1)
    collision = np.any(np.diff(days, axis=1) == 0, axis=1)
    approx = 1 - np.exp(-n * (n - 1) / 730)
    print(f"n={n:2d} | exact={exact(n):.4f} | exponential approx.={approx:.4f} | MC={collision.mean():.4f}")
