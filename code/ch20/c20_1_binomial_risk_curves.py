# Theory of Statistics — Chapter 20: Risk Curves for a Binomial Proportion and the Minimax Estimator

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

n = 16
p = np.linspace(0, 1, 401)
x = np.arange(n + 1)
pmf = stats.binom.pmf(x[:, None], n, p[None, :])          # (n+1) x len(p)

def risk(d):                                               # exact risk under squared loss
    return ((d[:, None] - p[None, :])**2 * pmf).sum(axis=0)

mle = x / n
a = np.sqrt(n) / 2
minimax = (x + a) / (n + 2 * a)                           # Bayes for Beta(sqrt(n)/2, sqrt(n)/2)
laplace = (x + 1) / (n + 2)                                # Bayes for Beta(1,1)
R_mle, R_mm, R_lap = risk(mle), risk(minimax), risk(laplace)

print(f"n = {n} | minimax risk (theory) 1/(4(1+sqrt n)^2) = {1/(4*(1+np.sqrt(n))**2):.6f}")
print(f"Risk of the minimax rule: min = {R_mm.min():.6f}, max = {R_mm.max():.6f}  (equaliser)")
print(f"Maximum risks: MLE = {R_mle.max():.6f} | Laplace = {R_lap.max():.6f} | minimax = {R_mm.max():.6f}")
better = p[R_mm < R_mle]
print(f"Minimax rule better than MLE: p in ({better.min():.3f}, {better.max():.3f})  (theory: 0.2, 0.8)")
print(f"Bayes risks under the uniform prior: MLE = {np.trapezoid(R_mle, p):.6f} | "
      f"Laplace = {np.trapezoid(R_lap, p):.6f} | minimax = {np.trapezoid(R_mm, p):.6f}")

fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(p, R_mle, label=r"MLE $X/n$")
ax.plot(p, R_lap, "--", label=r"Laplace $(X+1)/(n+2)$")
ax.plot(p, R_mm, lw=2, label=r"Minimax $(X+\sqrt{n}/2)/(n+\sqrt{n})$")
ax.set_xlabel("$p$"); ax.set_ylabel(r"$R(p,\delta)$"); ax.legend(fontsize=8)
ax.set_title(f"Risk functions under squared loss, n = {n}", fontsize=10)
fig.tight_layout(); fig.savefig("karar_binom_risk.pdf")
