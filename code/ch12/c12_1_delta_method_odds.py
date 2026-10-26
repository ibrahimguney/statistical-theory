# Theory of Statistics — Chapter 12: The Delta Method for the Odds

import numpy as np

rng = np.random.default_rng(42)
p, n = 0.3, 500
samples = rng.binomial(1, p, size=(100_000, n))
tau_hat = np.mean(samples, axis=1) / (1.0 - np.mean(samples, axis=1))
print(f"Delta theoretical variance: {(p/((1-p)**3))/n:.6f} | Empirical: {np.var(tau_hat, ddof=1):.6f}")
eta_hat = np.log(tau_hat)
print(f"Log-odds delta variance: {1/(p*(1-p)*n):.6f} | Empirical: {np.var(eta_hat, ddof=1):.6f}")
