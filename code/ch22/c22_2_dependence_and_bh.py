# Theory of Statistics — Chapter 22: FWER and FDR for Dependent Tests; Visualising the BH Procedure

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
m, alpha, M = 100, 0.05, 4000

def bh(p, alpha=alpha):
    m = len(p); s = np.argsort(p); ok = p[s] <= alpha * np.arange(1, m + 1) / m
    k = (np.nonzero(ok)[0].max() + 1) if ok.any() else 0
    rej = np.zeros(m, bool); rej[s[:k]] = True; return rej

print("All H0 true (m = 100); equicorrelated normal test statistics")
print(" rho | Bonferroni FWER | uncorrected FWER | BH FDR (= FWER)")
for rho in (0.0, 0.3, 0.6, 0.9):
    common = rng.standard_normal((M, 1))
    z = np.sqrt(rho) * common + np.sqrt(1 - rho) * rng.standard_normal((M, m))
    p = 2 * stats.norm.sf(np.abs(z))
    bon = (p <= alpha / m).any(axis=1).mean()
    unc = (p <= alpha).any(axis=1).mean()
    bhf = np.mean([bh(r).any() for r in p])
    print(f" {rho:.1f} |      {bon:.4f}     |      {unc:.4f}      |     {bhf:.4f}")

# Visual interpretation of BH: orderej p-values and the line alpha*k/m
has_eff = np.zeros(m, bool); has_eff[:15] = True
z = rng.standard_normal(m) + 3.0 * has_eff
p = stats.norm.sf(z); s = np.sort(p); k = np.arange(1, m + 1)
rej = bh(p); R = rej.sum()
print(f"Example data: BH rejected {R} hypotheses; real effects: {(rej & has_eff).sum()}, "
      f"false discoveries: {(rej & ~has_eff).sum()}. Bonferroni rejections: {(p <= alpha / m).sum()}.")
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.plot(k[:40], s[:40], "o", ms=3, label="orderej p-values $p_{(k)}$")
ax.plot(k[:40], alpha * k[:40] / m, "-", label=r"BH line $\alpha k/m$")
ax.axhline(alpha / m, ls=":", color="gray", label=r"Bonferroni $\alpha/m$")
ax.axvline(R + 0.5, ls="--", color="C3", lw=0.8)
ax.set_xlabel("$k$"); ax.set_ylabel("p-value"); ax.set_ylim(-0.002, 0.03); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("bh_dogrusu.pdf")
