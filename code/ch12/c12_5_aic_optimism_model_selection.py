# Theory of Statistics — Chapter 12: AIC: The Optimism of the Likelihood and Model Selection

import numpy as np

rng = np.random.default_rng(42)

# 1) Optimism: E[l_n(theta^; X) - l_n(theta^; Y)], Y an independent new sample.
#    Exponential model, data Gamma(a): theoretical limit tr(A^{-1}B) = 1/a; exact value n/(n a - 1).
n, R = 50, 1_000_000
for a in [2.0, 1.0, 0.5]:
    SX = rng.gamma(n * a, 1 / a, size=R)                   # sum: Gamma(n a), mean n
    SY = rng.gamma(n * a, 1 / a, size=R)                   # sum of the independent new sample
    lam = n / SX                                           # MLE
    l_X = n * np.log(lam) - lam * SX                       # the log-likelihood depends only on the sum
    l_Y = n * np.log(lam) - lam * SY
    print(f"a={a:3.1f} | optimism = {np.mean(l_X - l_Y):.3f} | exact = {n/(n*a - 1):.3f}"
          f" | tr(A^-1 B) = 1/a = {1/a:.3f} | k assumed by AIC = 1")

# 2) Choosing a polynomial degree: the true model is cubic, n = 40, sigma = 0.5
n, R, dmax = 40, 2000, 8
x = np.linspace(-1, 1, n)
mu = 1 + 2*x - 3*x**3
xt = np.linspace(-1, 1, 2000)
mut = 1 + 2*xt - 3*xt**3
choice = {"AIC": np.zeros(dmax + 1, int), "BIC": np.zeros(dmax + 1, int), "oracle": np.zeros(dmax + 1, int)}
score = {"AIC": 0.0, "BIC": 0.0, "oracle": 0.0}
for _ in range(R):
    y = mu + 0.5 * rng.normal(size=n)
    aic, bic, test = [], [], []
    for d in range(dmax + 1):
        V = np.vander(x, d + 1)
        b = np.linalg.lstsq(V, y, rcond=None)[0]
        s2 = np.mean((y - V @ b)**2)                       # MLE of sigma^2
        ll = -n / 2 * (np.log(2 * np.pi * s2) + 1)
        k = d + 2                                          # d + 1 coefficients and sigma^2
        aic.append(-2 * ll + 2 * k); bic.append(-2 * ll + np.log(n) * k)
        # oracle: expected log-likelihood at new points (knowing the true mu)
        mt = np.vander(xt, d + 1) @ b
        test.append(-0.5 * np.log(2 * np.pi * s2) - np.mean((mut - mt)**2 + 0.25) / (2 * s2))
    for name, d in [("AIC", np.argmin(aic)), ("BIC", np.argmin(bic)), ("oracle", np.argmax(test))]:
        choice[name][d] += 1
        score[name] += test[d] / R                          # log-likelihood of the chosen model on new data
print("chosen degree:   " + "  ".join(f"{d:4d}" for d in range(dmax + 1)))
for name, c in choice.items():
    print(f"{name:6s} (%)      " + "  ".join(f"{100*v/R:4.0f}" for v in c)
          + f"   | mean log-likelihood on new data (per observation) = {score[name]:.4f}")
