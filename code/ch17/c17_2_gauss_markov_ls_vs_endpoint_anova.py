# Theory of Statistics — Chapter 17: Gauss--Markov: LS versus the Endpoint Estimator; ANOVA

import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
n, R = 20, 200_000
x = np.arange(1, n + 1, dtype=float)
Y = 2.0 + 0.7*x + rng.normal(0, 1.5, (R, n))
xc = x - x.mean()
b_ls = (Y @ xc) / np.sum(xc**2)
b_end = (Y[:, -1] - Y[:, 0]) / (x[-1] - x[0])
print(f"LS slope  : mean={b_ls.mean():.4f}, Var={b_ls.var():.5f}")
print(f"Endpoints : mean={b_end.mean():.4f}, Var={b_end.var():.5f}")
print(f"Variance ratio = {b_end.var()/b_ls.var():.3f} | theory n(n+1)/(6(n-1)) = {n*(n+1)/(6*(n-1)):.3f}")

# One-way ANOVA: F by hand, the general C*beta formula and scipy
g = [rng.normal(m, 2.0, k) for m, k in [(10, 8), (11, 10), (13, 7)]]
y = np.concatenate(g); N, kgr = y.size, len(g)
ybar = y.mean()
F_hand = (sum(len(a)*(a.mean() - ybar)**2 for a in g)/(kgr - 1)) / (sum(((a - a.mean())**2).sum() for a in g)/(N - kgr))
Xg = np.zeros((N, kgr)); s = 0
for j, a in enumerate(g):
    Xg[s:s + len(a), j] = 1; s += len(a)
bh = np.linalg.solve(Xg.T @ Xg, Xg.T @ y); s2g = np.sum((y - Xg @ bh)**2)/(N - kgr)
Cg = np.array([[1, -1, 0], [1, 0, -1]])
d = Cg @ bh
F_gen = d @ np.linalg.solve(Cg @ np.linalg.inv(Xg.T @ Xg) @ Cg.T, d) / 2 / s2g
print(f"ANOVA F: by hand={F_hand:.4f} | general linear hypothesis={F_gen:.4f} | scipy f_oneway={stats.f_oneway(*g).statistic:.4f}"
      f" | p={stats.f_oneway(*g).pvalue:.4f}")
