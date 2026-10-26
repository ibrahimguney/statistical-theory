# Theory of Statistics — Chapter 12: ARE, Variance Stabilisation and the Second-Order Delta Method

import numpy as np

rng = np.random.default_rng(42)
n, R = 101, 100_000
for name, X in [("Normal ", rng.normal(size=(R, n))),
              ("Laplace", rng.laplace(size=(R, n))),
              ("t_3    ", rng.standard_t(3, size=(R, n)))]:
    vm, vmed = X.mean(axis=1).var(), np.median(X, axis=1).var()
    print(f"{name} | n*Var(mean)={n*vm:.3f} | n*Var(median)={n*vmed:.3f} | ARE(median, mean)={vm/vmed:.3f}")

print("Poisson: Var(2*sqrt(Xbar)) * n  (target 1, independent of lambda)")
for lam in [1, 5, 50]:
    Xb = rng.poisson(lam, size=(R, 200)).mean(axis=1)
    print(f"  lambda={lam:3d} | n*Var(Xbar)={200*Xb.var():7.3f} | n*Var(2 sqrt(Xbar))={200*np.var(2*np.sqrt(Xb)):.3f}")

n = 400
pb = rng.binomial(n, 0.5, size=R) / n
W = n * (pb*(1 - pb) - 0.25)
print(f"Second-order delta: E[W]={W.mean():.4f} (theory -0.25), Var[W]={W.var():.4f} (theory 2/16={2/16:.4f})")
