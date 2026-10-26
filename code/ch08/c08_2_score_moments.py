# Theory of Statistics — Chapter 8: Moments of the Score: Checking the Information Identity

import numpy as np

rng = np.random.default_rng(42)
N = 2_000_000
# Gamma(known shape a, rate b): ln f = a ln b - ln G(a) + (a-1) ln x - b x
a, b = 3.0, 2.0
X = rng.gamma(a, 1/b, size=N)
score = a / b - X              # d/db ln f
hess = -a / b**2 * np.ones(N) # d2/db2 ln f
print(f"Gamma : E[s]={score.mean():+.4f} | Var(s)={score.var():.4f} | -E[s']={-hess.mean():.4f} | theory I={a/b**2:.4f}")

# Cauchy location: s = 2(x-t)/(1+(x-t)^2), I = 1/2
Y = rng.standard_cauchy(N)
s = 2*Y / (1 + Y**2)
h = (2*Y**2 - 2) / (1 + Y**2)**2     # d/dt s
print(f"Cauchy: E[s]={s.mean():+.4f} | Var(s)={s.var():.4f} | -E[s']={-h.mean():.4f} | theory I=0.5000")

# Uniform(0, theta): regularity violated, E[s] != 0
theta = 2.0
print(f"Uniform: E[s] = -1/theta = {-1/theta:.3f}  (not zero!)")
