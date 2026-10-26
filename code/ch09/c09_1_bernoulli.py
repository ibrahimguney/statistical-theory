# Theory of Statistics — Chapter 9: Bernoulli: X_1 -> bar X

import numpy as np

rng = np.random.default_rng(42)
samples = rng.binomial(1, 0.4, size=(200_000, 10))
print(f"Var(W=X_1): {np.var(samples[:,0], ddof=1):.4f} | Var(W*=X_bar): {np.var(np.mean(samples, axis=1), ddof=1):.4f}")
print(f"Theory    : {0.4*0.6:.4f}                | {0.4*0.6/10:.4f}")
