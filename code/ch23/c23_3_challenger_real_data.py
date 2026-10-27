# Theory of Statistics — Chapter 23: Real Data: O-Ring Damage Before Challenger

import numpy as np
from scipy import stats

# Dalal, Fowlkes and Hoadley (1989): for the 23 flights before Challenger (flight 4, not recovered, excluded)
# the launch temperature (Fahrenheit) and whether at least one O-ring had thermal damage. Source: R package vcd.
flight = [
    '1', '2', '3', '5', '6', '7', '8', '9', '41B', '41C', '41D', '41G', '51A', '51C', '51D',
    '51B', '51G', '51F', '51I', '51J', '61A', '61C', '61I',
]
temperature = np.array([
    66, 70, 69, 68, 67, 72, 73, 70, 57, 63, 70, 78, 67, 53, 67, 75, 70, 81, 76, 79, 75, 58, 76,
], dtype=float)
damage = np.array([
    0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0,
], dtype=float)

def logistic(X, y):                                        # IRLS (Fisher scoring)
    b = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ b))
        I = X.T @ ((p * (1 - p))[:, None] * X)
        b = b + np.linalg.solve(I, X.T @ (y - p))
    p = 1 / (1 + np.exp(-X @ b))
    I = X.T @ ((p * (1 - p))[:, None] * X)
    ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return b, np.linalg.inv(I), ll

X = np.column_stack([np.ones_like(temperature), temperature])
b, V, ll = logistic(X, damage)
se = np.sqrt(np.diag(V))
print(f"{len(damage)} flights, {int(damage.sum())} with damage | beta^ = ({b[0]:.3f}, {b[1]:.4f}), SE = ({se[0]:.3f}, {se[1]:.4f})")
p0 = damage.mean()
ll0 = np.sum(damage * np.log(p0) + (1 - damage) * np.log(1 - p0))
G2 = 2 * (ll - ll0)
print(f"H0: beta1 = 0 | Wald z = {b[1]/se[1]:.3f}, p = {2*stats.norm.sf(abs(b[1]/se[1])):.4f} | LRT G^2 = {G2:.3f}, p = {stats.chi2.sf(G2, 1):.4f}")
print(f"odds multiplier for a 10 F drop = {np.exp(-10*b[1]):.2f}, 95% CI = [{np.exp(-10*(b[1] + 1.96*se[1])):.2f}, {np.exp(-10*(b[1] - 1.96*se[1])):.2f}]")
expit = lambda e: 1 / (1 + np.exp(-e))
for t in [75, 65, 53, 31]:
    x = np.array([1.0, t])
    eta, s = x @ b, np.sqrt(x @ V @ x)                     # delta method: SE on the logit scale
    print(f"T = {t} F: P(damage) = {expit(eta):.4f}, 95% CI = [{expit(eta - 1.96*s):.4f}, {expit(eta + 1.96*s):.4f}]")
