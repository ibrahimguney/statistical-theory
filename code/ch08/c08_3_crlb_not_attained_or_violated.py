# Theory of Statistics — Chapter 8: When the CRLB Is Not Attained and When It Is Violated

import numpy as np

rng = np.random.default_rng(42)
R = 300_000
lam = 2.0
print("Exp(lambda): lambda_tilde = (n-1)/sum X")
for n in [5, 20, 100]:
    T = rng.exponential(1/lam, size=(R, n)).sum(axis=1)
    lt = (n - 1) / T
    print(f"  n={n:3d} | mean={lt.mean():.4f} | Var={lt.var():.5f} | theory {lam**2/(n-2):.5f} | CRLB {lam**2/n:.5f}")

theta = 1.0
print("Uniform(0,theta): (n+1)/n * max  vs  'pseudo-CRLB' theta^2/n")
for n in [5, 20, 100]:
    t = (n + 1) / n * rng.uniform(0, theta, size=(R, n)).max(axis=1)
    print(f"  n={n:3d} | Var={t.var():.6f} | theory {theta**2/(n*(n+2)):.6f} | pseudo-CRLB {theta**2/n:.6f}")
