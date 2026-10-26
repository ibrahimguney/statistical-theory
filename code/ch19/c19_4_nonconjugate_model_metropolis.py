# Theory of Statistics — Chapter 19: A Non-Conjugate Model: The Metropolis Algorithm

import numpy as np
from scipy import stats, integrate

rng = np.random.default_rng(42)
x, n, s0 = 10, 50, 1.5
def log_post(eta):        # log-likelihood + log-prior (up to constants)
    return x*eta - n*np.log1p(np.exp(eta)) - eta**2 / (2*s0**2)

N, s = 60_000, 0.6
chain = np.empty(N); eta = 0.0; lp = log_post(eta); acc = 0
for i in range(N):
    prop = eta + s*rng.standard_normal()
    lp_prop = log_post(prop)
    if np.log(rng.random()) < lp_prop - lp:
        eta, lp = prop, lp_prop; acc += 1
    chain[i] = eta
theta = 1/(1 + np.exp(-chain[5_000:]))          # discard burn-in

# Check: exact posterior mean by numerical integration
Z = integrate.quad(lambda e: np.exp(log_post(e)), -10, 10)[0]
m_exact = integrate.quad(lambda e: np.exp(log_post(e))/(1 + np.exp(-e)), -10, 10)[0] / Z
print(f"Acceptance rate = {acc/N:.3f}")
print(f"MCMC  posterior mean E[theta|x] = {theta.mean():.4f} | 95% interval = {np.percentile(theta, [2.5, 97.5]).round(4)}")
print(f"Exact posterior mean (quad)      = {m_exact:.4f}")
print(f"Comparison: Beta(2,18) conjugate posterior mean = {12/70:.4f} | MLE = {x/n:.4f}")
