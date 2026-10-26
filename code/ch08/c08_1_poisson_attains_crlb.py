# Theory of Statistics — Chapter 8: Poisson: bar X Attains the CRLB

import numpy as np

rng = np.random.default_rng(42)
lam = 3.0
for n in [5, 15, 30, 100]:
    samples = rng.poisson(lam=lam, size=(150_000, n))
    var_x = np.var(np.mean(samples, axis=1), ddof=1)
    print(f"n = {n:3d} | CRLB: {lam/n:.6f} | Empirical Var: {var_x:.6f}")
