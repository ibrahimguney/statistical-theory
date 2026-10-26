# Theory of Statistics — Chapter 3: Verifying the Cauchy--Schwarz Inequality

import numpy as np

rng = np.random.default_rng(42)
cov = [[4.0, 0.85 * 2.0 * 3.0], [0.85 * 2.0 * 3.0, 9.0]]
samples = rng.multivariate_normal([2.0, -1.0], cov, size=500_000)
X, Y = samples[:, 0], samples[:, 1]
print(f"(E[XY])^2: {(np.mean(X*Y))**2:.4f} <= E[X^2]E[Y^2]: {np.mean(X**2)*np.mean(Y**2):.4f}")

# Equality case: Y = a + bX  =>  |Corr| = 1 ; equality for the centred variables
Y_lin = 3.0 - 2.0 * X
Xc, Yc = X - X.mean(), Y_lin - Y_lin.mean()
print(f"Linear relation: (E[XcYc])^2 = {np.mean(Xc*Yc)**2:.4f} , E[Xc^2]E[Yc^2] = {np.mean(Xc**2)*np.mean(Yc**2):.4f}")
print(f"Sample correlation (for rho=0.85): {np.corrcoef(X, Y)[0,1]:.4f}")
