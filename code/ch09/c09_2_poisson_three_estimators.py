# Theory of Statistics — Chapter 9: Poisson: Three Estimators of P(X=0)

import numpy as np

rng = np.random.default_rng(42)
lam, n, R = 2.0, 10, 400_000
X = rng.poisson(lam, size=(R, n))
T = X.sum(axis=1)
target = np.exp(-lam)
estimators = {"W  = I(X1=0)        ": (X[:, 0] == 0).astype(float),
             "W* = (1-1/n)^T (RB) ": (1 - 1/n)**T,
             "exp(-Xbar) (plug-in)": np.exp(-T/n)}
print(f"Target e^-lambda = {target:.5f} | CRLB = {np.exp(-2*lam)*lam/n:.3e}")
for name, t in estimators.items():
    print(f"{name} | mean={t.mean():.5f} | bias={t.mean()-target:+.5f} | Var={t.var():.3e} | MSE={np.mean((t-target)**2):.3e}")
print(f"Theory Var(W*) = {np.exp(-2*lam)*(np.exp(lam/n)-1):.3e}")
