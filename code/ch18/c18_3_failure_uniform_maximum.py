# Theory of Statistics — Chapter 18: Failure: The Maximum of Uniform(0,theta)

import numpy as np

rng = np.random.default_rng(42)
theta, n, B = 1.0, 50, 20_000
x = rng.uniform(0, theta, n)
m = x.max()
np_boot = x[rng.integers(0, n, (B, n))].max(1)                 # nonparametric
p_boot = rng.uniform(0, m, (B, n)).max(1)                       # parametric
true_max = rng.uniform(0, theta, (B, n)).max(1)
print(f"P*(X*_(n) = X_(n)) = {np.mean(np_boot == m):.4f} | theory 1-(1-1/n)^n = {1-(1-1/n)**n:.4f}")
for name, s, ref in [("True distribution   ", n*(theta - true_max)/theta, None),
                   ("Nonparametric       ", n*(m - np_boot)/m, None),
                   ("Parametric bootstrap", n*(m - p_boot)/m, None)]:
    q = np.percentile(s, [25, 50, 75])
    print(f"{name} | n(theta-max)/theta: P(=0)={np.mean(s == 0):.3f} | quartiles={np.round(q, 3)} (Exp(1): [0.288 0.693 1.386])")
