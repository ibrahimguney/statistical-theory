# Theory of Statistics — Chapter 4: Multivariate Normal: Cholesky Generation, Conditional Distribution, Mahalanobis

import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
mu = np.array([170.0, 70.0, 40.0])
sd = np.array([10.0, 12.0, 8.0])
R = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.5], [0.3, 0.5, 1.0]])
Sigma = np.outer(sd, sd) * R
L = np.linalg.cholesky(Sigma)
N = 1_000_000
X = mu + rng.normal(size=(N, 3)) @ L.T
print("Sample covariance - Sigma (largest absolute difference):", np.abs(np.cov(X.T) - Sigma).max().round(3))

# Conditional distribution of X1 | (X2, X3): theoretical and empirical via "slicing"
x2 = np.array([80.0, 45.0])
S12, S22 = Sigma[0, 1:], Sigma[1:, 1:]
m_cond = mu[0] + S12 @ np.linalg.solve(S22, x2 - mu[1:])
v_cond = Sigma[0, 0] - S12 @ np.linalg.solve(S22, S12)
sel = (np.abs(X[:, 1] - x2[0]) < 1.0) & (np.abs(X[:, 2] - x2[1]) < 1.0)
print(f"E[X1|x2,x3]: theory={m_cond:.3f}, slice={X[sel, 0].mean():.3f} | sd: theory={np.sqrt(v_cond):.3f}, slice={X[sel, 0].std():.3f} (n={sel.sum()})")

# Squared Mahalanobis distance ~ chi2_3
D2 = np.einsum("ij,jk,ik->i", X - mu, np.linalg.inv(Sigma), X - mu)
print(f"Mahalanobis D^2: mean={D2.mean():.4f} (3) | P(D^2 > chi2_3,0.05)={np.mean(D2 > stats.chi2.ppf(0.95, 3)):.4f} (0.05)")
