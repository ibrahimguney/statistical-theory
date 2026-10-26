# Theory of Statistics — Chapter 7: Estimators for Uniform(0,theta)

import numpy as np

rng = np.random.default_rng(42)
theta, R = 10.0, 100_000
print(" n   | MSE(2*Xbar)  theory   | MSE(X_max)  theory   | MSE((n+1)/n X_max) theory")
for n in [5, 20, 100]:
    X = rng.uniform(0, theta, size=(R, n))
    t1, t2 = 2*X.mean(axis=1), X.max(axis=1)
    t3 = (n + 1) / n * t2
    m = lambda t: np.mean((t - theta)**2)
    print(f"{n:4d} | {m(t1):8.4f} {theta**2/(3*n):8.4f} | {m(t2):8.4f} {2*theta**2/((n+1)*(n+2)):8.4f}"
          f" | {m(t3):8.4f} {theta**2/(n*(n+2)):8.4f}")
