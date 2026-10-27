# Theory of Statistics — Chapter 12: Misspecified Model: Sandwich Variance and the Huber Estimator

import numpy as np
from scipy import stats, integrate

rng = np.random.default_rng(42)

# 1) Exponential model, true data Gamma(shape 1/2, scale 2): mean 1, variance 2.
#    Pseudo-true value lambda* = 1/mu = 1; sandwich variance sigma^2/mu^4 = 2, model 1/I = 1.
R = 20_000
for n in [50, 400]:
    X = rng.gamma(0.5, 2.0, size=(R, n))
    lam = 1 / X.mean(axis=1)
    se_model = lam / np.sqrt(n)                            # 1/sqrt(n I_1(lam))
    A = 1 / lam**2                                         # -mean dpsi/dlambda
    B = np.mean((1 / lam[:, None] - X)**2, axis=1)         # mean psi^2
    se_sandwich = np.sqrt(B / A**2 / n)
    cov_m = np.mean(np.abs(lam - 1) <= 1.96 * se_model)
    cov_s = np.mean(np.abs(lam - 1) <= 1.96 * se_sandwich)
    print(f"Exponential model, n={n:3d} | n*Var(lam)={n*lam.var():.3f} | model n*SE^2={np.mean(n*se_model**2):.3f}"
          f" | sandwich n*SE^2={np.mean(n*se_sandwich**2):.3f} | coverage: model={cov_m:.3f}, sandwich={cov_s:.3f}")

# 2) Huber location estimator (c = 1.345), data t_3; comparison with the mean and the median
c, n, R = 1.345, 200, 20_000
psi = lambda u: np.clip(u, -c, c)
def huber(X):
    th = np.median(X, axis=1)
    for _ in range(50):                                    # vectorised Newton steps
        u = X - th[:, None]
        th = th + psi(u).mean(axis=1) / np.maximum(np.mean(np.abs(u) <= c, axis=1), 1e-12)
    return th
t3 = stats.t(3)
A = t3.cdf(c) - t3.cdf(-c)                                 # E psi'
B = integrate.quad(lambda u: psi(u)**2 * t3.pdf(u), -np.inf, np.inf)[0]   # E psi^2
X = t3.rvs(size=(R, n), random_state=rng)
for name, T in [("mean", X.mean(axis=1)), ("median", np.median(X, axis=1)), ("Huber", huber(X))]:
    print(f"t_3, n={n} | {name:8s} | n*Var={n*T.var():.3f}")
print(f"Huber theoretical B/A^2 = {B/A**2:.3f} | mean theoretical = 3.000 | median theoretical = {1/(4*t3.pdf(0)**2):.3f}")
