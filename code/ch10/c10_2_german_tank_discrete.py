# Theory of Statistics — Chapter 10: The German Tank Problem: Discrete Version

import numpy as np

rng = np.random.default_rng(42)
N, k, R = 250, 8, 200_000
m = np.array([rng.choice(N, k, replace=False).max() + 1 for _ in range(R)])
umvue = m * (1 + 1/k) - 1
print(f"m(1+1/k)-1 : mean={umvue.mean():.2f} (target {N}) | std={umvue.std():.2f}")
print(f"2*mean - 1 : ", end="")
o = np.array([rng.choice(N, k, replace=False).mean() + 1 for _ in range(R)])
print(f"mean={(2*o - 1).mean():.2f} | std={(2*o - 1).std():.2f}")
print(f"Theory std(UMVUE) = {np.sqrt((N - k)*(N + 1)/(k*(k + 2))):.2f}")
