# Theory of Statistics — Chapter 13: A Randomised Poisson Test

import numpy as np
from scipy import stats

alpha, n, l0, l1 = 0.05, 10, 1.0, 2.0
T0, T1 = stats.poisson(n*l0), stats.poisson(n*l1)
c = int(T0.ppf(1 - alpha))            # smallest c: P0(T > c) <= alpha
while T0.sf(c) > alpha: c += 1
gamma = (alpha - T0.sf(c)) / T0.pmf(c)
print(f"Reject: T > {c} ; if T = {c} reject with probability gamma = {gamma:.4f}")
print(f"P0(T >= {c+1}) = {T0.sf(c):.4f} | P0(T >= {c}) = {T0.sf(c-1):.4f}")
print(f"Size = {T0.sf(c) + gamma*T0.pmf(c):.4f}")
print(f"Power (randomised)        = {T1.sf(c) + gamma*T1.pmf(c):.4f}")
print(f"Power (T > {c}, conservative) = {T1.sf(c):.4f}")

rng = np.random.default_rng(42)
T = rng.poisson(n*l1, size=1_000_000)
rej = (T > c) | ((T == c) & (rng.random(T.size) < gamma))
print(f"Power (simulation)        = {rej.mean():.4f}")
