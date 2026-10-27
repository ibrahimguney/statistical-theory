# Theory of Statistics — Chapter 13: The Sequential Probability Ratio Test (SPRT)

import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
# H0: N(0, 1), H1: N(1, 1); log-likelihood ratio increment Z = X - 1/2
alpha, beta = 0.05, 0.05
a, b = np.log(beta / (1 - alpha)), np.log((1 - beta) / alpha)   # Wald's boundaries
print(f"boundaries: a = {a:.3f}, b = {b:.3f}")

def sprt(mu, R=100_000, steps=150):
    X = rng.normal(mu, 1, size=(R, steps))
    S = np.cumsum(X - 0.5, axis=1)
    out = (S >= b) | (S <= a)
    N = out.argmax(axis=1) + 1                          # first exit time (always exits within steps)
    assert out.any(axis=1).all()
    rej = S[np.arange(R), N - 1] >= b
    return N, rej

KL = 0.5                                                   # KL(f0||f1) = KL(f1||f0) = 1/2
for mu, name in [(0.0, "H0"), (1.0, "H1")]:
    N, rej = sprt(mu)
    if mu == 0:
        err, appr = rej.mean(), (alpha * b + (1 - alpha) * a) / (-KL)
        print(f"{name}: type I error = {err:.4f} (target {alpha}, Wald bound alpha/(1-beta) = {alpha/(1-beta):.4f})")
    else:
        err, appr = 1 - rej.mean(), ((1 - beta) * b + beta * a) / KL
        print(f"{name}: type II error = {err:.4f} (target {beta}, Wald bound beta/(1-alpha) = {beta/(1-alpha):.4f})")
    print(f"    E[N] = {N.mean():.2f} | Wald approximation (ignoring overshoot) = {appr:.2f}"
          f" | median N = {np.median(N):.0f} | P(N > 11) = {np.mean(N > 11):.3f} | max N = {N.max()}")
# Fixed-sample test with the same error probabilities (Example: Sample Size Calculation)
n_fixed = int(np.ceil(((stats.norm.ppf(1 - alpha) + stats.norm.ppf(1 - beta)) / 1.0)**2))
print(f"n required by the fixed-sample test = {n_fixed}")
