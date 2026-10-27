# Theory of Statistics — Chapter 19: Empirical Bayes: James--Stein and Robbins' Formula

import numpy as np

rng = np.random.default_rng(42)

# 1) Hierarchical normal model: theta_i ~ N(mu, tau^2), X_i | theta_i ~ N(theta_i, 1), i = 1..k.
#    Average total loss (Bayes risk): X, JS shrinking to 0, Efron--Morris shrinking to the mean, oracle Bayes
k, R = 10, 100_000
for mu, tau2 in [(0.0, 0.5), (0.0, 2.0), (0.0, 8.0), (3.0, 2.0)]:
    theta = mu + np.sqrt(tau2) * rng.normal(size=(R, k))
    X = theta + rng.normal(size=(R, k))
    B = 1 / (1 + tau2)
    js = (1 - (k - 2) / np.sum(X**2, axis=1, keepdims=True)) * X
    xbar = X.mean(axis=1, keepdims=True)
    S = np.sum((X - xbar)**2, axis=1, keepdims=True)
    em = xbar + (1 - (k - 3) / S) * (X - xbar)
    bayes = mu + (1 - B) * (X - mu)                        # oracle that knows mu and tau^2
    loss = lambda d: np.mean(np.sum((d - theta)**2, axis=1))
    js_theory = f"{k-(k-2)*B:6.3f}" if mu == 0 else "   -  "     # formula only for mu = 0
    print(f"mu={mu:.0f}, tau^2={tau2:3.1f} | X={loss(X):6.3f} | JS={loss(js):6.3f}"
          f" (theory {js_theory}) | EM={loss(em):6.3f} (theory {k-(k-3)*B:6.3f})"
          f" | oracle={loss(bayes):6.3f} (theory {k*(1-B):6.3f})")

# 2) Robbins' formula: theta_i ~ Gamma(shape 2, rate 1), X_i | theta_i ~ Poisson(theta_i), N units.
#    True posterior mean (Gamma--Poisson): (2 + x) / 2
N = 2000
theta = rng.gamma(2.0, 1.0, size=N)
X = rng.poisson(theta)
counts = np.bincount(X, minlength=12)
print(" x | count | Robbins (x+1)N(x+1)/N(x) | true Bayes | MLE")
for x in range(8):
    robbins = (x + 1) * counts[x + 1] / counts[x]
    print(f"{x:2d} | {counts[x]:5d} | {robbins:24.3f} | {(2 + x) / 2:10.3f} | {x:3d}")
# mean squared error over the units with x <= 7 (most of the data)
s = X <= 7
rob = np.array([(x + 1) * counts[x + 1] / counts[x] for x in range(8)])[X[s]]
err = lambda d: np.mean((d - theta[s])**2)
print(f"Mean squared error over the {s.sum()} units with x<=7: MLE={err(X[s]):.3f} | Robbins={err(rob):.3f}"
      f" | true Bayes={err((2 + X[s]) / 2):.3f}")
