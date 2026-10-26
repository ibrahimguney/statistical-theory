# Theory of Statistics — Chapter 16: The Pivotal Interval for Uniform(0,theta)

import numpy as np

rng = np.random.default_rng(42)
theta, a, R = 7.0, 0.05, 200_000
for n in [5, 20, 100]:
    M = rng.uniform(0, theta, size=(R, n)).max(axis=1)
    L, U = M, M * a**(-1/n)
    print(f"n={n:3d} | coverage={np.mean((L <= theta) & (theta <= U)):.4f} | mean length={np.mean(U - L):.4f}"
          f" | theory={theta*n/(n+1)*(a**(-1/n) - 1):.4f}")
