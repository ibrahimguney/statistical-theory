# Theory of Statistics — Chapter 16: Binomial Proportion: Wald, Wilson, Clopper--Pearson, Agresti--Coull

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def intervals(x, n, a=0.05):
    z = stats.norm.ppf(1 - a/2); p = x / n
    wald = (p - z*np.sqrt(p*(1-p)/n), p + z*np.sqrt(p*(1-p)/n))
    m = (p + z**2/(2*n)) / (1 + z**2/n); h = z/(1 + z**2/n)*np.sqrt(p*(1-p)/n + z**2/(4*n**2))
    wilson = (m - h, m + h)
    cp = (np.nan_to_num(stats.beta.ppf(a/2, x, n - x + 1), nan=0.0),
          np.nan_to_num(stats.beta.ppf(1 - a/2, x + 1, n - x), nan=1.0))
    nt = n + z**2; pt = (x + z**2/2)/nt
    ac = (pt - z*np.sqrt(pt*(1-pt)/nt), pt + z*np.sqrt(pt*(1-pt)/nt))
    return {"Wald": wald, "Wilson": wilson, "Clopper-Pearson": cp, "Agresti-Coull": ac}

def coverage(n, ps):
    x = np.arange(n + 1); A = intervals(x, n)
    return {k: np.array([np.sum(stats.binom.pmf(x, n, p) * ((L <= p) & (p <= U))) for p in ps])
            for k, (L, U) in A.items()}

ps = np.linspace(0.005, 0.995, 991)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, n in zip(axes, [30, 100]):
    K = coverage(n, ps)
    for k, v in K.items():
        ax.plot(ps, v, lw=0.8, label=k)
        print(f"n={n:3d} {k:16s} | mean coverage={v.mean():.4f} | min coverage={v.min():.4f}")
    ax.axhline(0.95, color="k", ls=":"); ax.set_ylim(0.75, 1.0); ax.set_xlabel("p"); ax.set_title(f"n = {n}")
axes[0].set_ylabel("coverage probability"); axes[1].legend(fontsize=8, loc="lower center")
fig.tight_layout(); fig.savefig("binom_kapsama.pdf")
for p in [0.01, 0.05, 0.2, 0.5]:
    K = coverage(100, [p])
    print(f"n=100, p={p:.2f} | Wald={K['Wald'][0]:.4f} | Wilson={K['Wilson'][0]:.4f} | CP={K['Clopper-Pearson'][0]:.4f}")
