# Theory of Statistics — Chapter 21: Kernel Density Estimation: Bandwidth and the MISE Rate

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
def f(x):                                         # bimodal mixture
    return 0.6 * stats.norm.pdf(x, -1.5, 0.7) + 0.4 * stats.norm.pdf(x, 1.5, 0.5)
def gen(n):
    k = rng.random(n) < 0.6
    return np.where(k, rng.normal(-1.5, 0.7, n), rng.normal(1.5, 0.5, n))
def kde(x, data, h):                              # Gaussian-kernel estimator
    return stats.norm.pdf((x[:, None] - data[None, :]) / h).mean(axis=1) / h

grid = np.linspace(-5, 5, 801); dx = grid[1] - grid[0]
# R(f'') numerically: AMISE-optimal h = [R(K) / (mu2^2 R(f'') n)]^(1/5); Gaussian kernel: R(K)=1/(2 sqrt(pi)), mu2=1
f2 = np.gradient(np.gradient(f(grid), dx), dx)
Rf2 = (f2**2).sum() * dx
def h_opt(n): return (1 / (2 * np.sqrt(np.pi)) / (Rf2 * n))**0.2

print(f"R(f'') = {Rf2:.4f}")
print("    n | h_opt  | MISE (MC) | AMISE theory")
res = []
for n in (100, 400, 1600, 6400):
    h = h_opt(n)
    mise = np.mean([((kde(grid, gen(n), h) - f(grid))**2).sum() * dx for _ in range(100)])
    amise = 1.25 * Rf2**0.2 * (1 / (2 * np.sqrt(np.pi)))**0.8 * n**-0.8
    res.append((n, mise)); print(f"{n:5d} | {h:.4f} | {mise:.6f}  | {amise:.6f}")
sl = np.polyfit(np.log([s[0] for s in res]), np.log([s[1] for s in res]), 1)[0]
print(f"slope of log MISE ~ log n = {sl:.3f} (theory: -0.8; parametric rate: -1)")

data = gen(200)
h_silv = 1.06 * data.std(ddof=1) * 200**-0.2
print(f"n=200: Silverman h = {h_silv:.3f} | AMISE-optimal h = {h_opt(200):.3f}")
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(grid, f(grid), "k", lw=2, label="true $f$")
for h, st in ((0.08, ":"), (h_opt(200), "-"), (h_silv, "--"), (1.0, "-.")):
    ax.plot(grid, kde(grid, data, h), st, label=f"h = {h:.2f}")
ax.set_xlim(-4.5, 4); ax.legend(fontsize=8); ax.set_xlabel("$x$")
fig.tight_layout(); fig.savefig("cekirdek_bant.pdf")
