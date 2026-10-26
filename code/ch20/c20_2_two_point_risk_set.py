# Theory of Statistics — Chapter 20: The Risk Set in a Two-Point Problem: Bayes and Minimax Rules

import itertools
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

n, t0, t1 = 5, 0.3, 0.6
x = np.arange(n + 1)
f0, f1 = stats.binom.pmf(x, n, t0), stats.binom.pmf(x, n, t1)

# 0-1 loss: R0 = P_t0(decide theta1), R1 = P_t1(decide theta0). 2^6 non-randomised rules.
points = []
for A in itertools.product([0, 1], repeat=n + 1):          # A[x]=1: decide theta1
    A = np.array(A)
    points.append((f0[A == 1].sum(), f1[A == 0].sum()))
points = np.array(points)

pi = np.linspace(0, 1, 100_001)                            # pi = P(theta = theta0)
bayes = np.minimum(pi[:, None] * f0, (1 - pi[:, None]) * f1).sum(axis=1)
j = bayes.argmax()
print(f"Number of non-randomised rules: {len(points)}")
print(f"Smallest max. risk among non-randomised rules = {points.max(axis=1).min():.4f}")
print(f"Least favourable prior pi* = {pi[j]:.4f} | minimax risk = max_pi r(pi) = {bayes[j]:.4f}")

# Bayes rule at pi*: theta1 if LR = f1/f0 > pi*/(1-pi*); randomise at equality
lr = f1 / f0; thr = pi[j] / (1 - pi[j])
strict = lr > thr + 1e-9; bnd = np.isclose(lr, thr, atol=1e-3)
gamma = (f1[~strict].sum() - f0[strict].sum()) / (f0[bnd].sum() + f1[bnd].sum())
R0 = f0[strict].sum() + gamma * f0[bnd].sum()
R1 = f1[~strict & ~bnd].sum() + (1 - gamma) * f1[bnd].sum()
print(f"LR threshold = {thr:.4f}, boundary x = {x[bnd]}, randomisation probability gamma = {gamma:.4f}")
print(f"Minimax rule: R0 = {R0:.4f}, R1 = {R1:.4f} (equal error probabilities)")

fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.scatter(*points.T, s=10, color="gray", label="non-randomised rules")
from scipy.spatial import ConvexHull
h = ConvexHull(points); v = np.append(h.vertices, h.vertices[0])
ax.plot(*points[v].T, lw=0.8, color="C0", label="boundary of the risk set")
ax.plot([0, 1], [0, 1], ":", color="k", lw=0.8)
ax.plot(R0, R1, "o", color="C3", ms=7, label="minimax rule")
ax.set_xlabel(r"$R(\theta_0,\delta)$"); ax.set_ylabel(r"$R(\theta_1,\delta)$")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=7, loc="upper right")
fig.tight_layout(); fig.savefig("karar_risk_kumesi.pdf")
