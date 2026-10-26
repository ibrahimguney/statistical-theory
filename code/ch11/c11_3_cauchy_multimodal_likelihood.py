# Theory of Statistics — Chapter 11: Multimodality of the Cauchy Likelihood

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize

x = np.array([-4.2, -3.1, 0.4, 5.6, 6.3])
ll = lambda t: -np.sum(np.log1p((x - t)**2))
grid = np.linspace(-10, 12, 4001)
L = np.array([ll(t) for t in grid])
peaks = grid[1:-1][(L[1:-1] > L[:-2]) & (L[1:-1] > L[2:])]
print("Local maxima:", np.round(peaks, 3))
for t0 in [-5.0, 0.0, 6.0, np.median(x), grid[np.argmax(L)]]:   # the last two: the median and the best grid point
    r = minimize(lambda t: -ll(t[0]), [t0], method="BFGS")
    print(f"start={t0:5.2f} -> theta_hat={r.x[0]:7.4f}, logL={-r.fun:.4f}")

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(grid, L, "k-")
ax.plot(x, np.full_like(x, L.min()), "r|", ms=15, label="observations")
ax.set_xlabel(r"$\theta$"); ax.set_ylabel(r"$\ell(\theta)$ (+ const.)"); ax.legend()
fig.tight_layout(); fig.savefig("cauchy_ll.pdf")
