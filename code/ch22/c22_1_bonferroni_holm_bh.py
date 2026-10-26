# Theory of Statistics — Chapter 22: Bonferroni, Holm and Benjamini--Hochberg: FWER, FDR and Power

import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
m, m1, eff, alpha, M = 1000, 100, 3.0, 0.05, 2000

def bonferroni(p): return p <= alpha / len(p)
def holm(p):
    m = len(p); s = np.argsort(p); thr = alpha / (m - np.arange(m))
    ok = p[s] <= thr
    k = m if ok.all() else np.argmin(ok)            # reject up to the first failure
    rej = np.zeros(m, bool); rej[s[:k]] = True; return rej
def bh(p):
    m = len(p); s = np.argsort(p); ok = p[s] <= alpha * np.arange(1, m + 1) / m
    k = (np.nonzero(ok)[0].max() + 1) if ok.any() else 0
    rej = np.zeros(m, bool); rej[s[:k]] = True; return rej
def uncorrected(p): return p <= alpha

methods = {"Uncorrected": uncorrected, "Bonferroni": bonferroni, "Holm": holm, "BH": bh}
true_eff = np.zeros(m, bool); true_eff[:m1] = True             # the first m1 hypotheses are false (effect present)
res = {k: [] for k in methods}
for _ in range(M):
    z = rng.standard_normal(m) + eff * true_eff
    p = stats.norm.sf(z)                                   # one-sided p-values
    for name, f in methods.items():
        rej = f(p); V = (rej & ~true_eff).sum(); R = rej.sum()
        res[name].append((V >= 1, V / max(R, 1), (rej & true_eff).sum() / m1, R))
print(f"m = {m}, number of false H0 m1 = {m1}, effect = {eff}, alpha = {alpha}, replications = {M}")
print("Method       |  FWER  |  FDR   | power  | mean rej.")
for name, s in res.items():
    s = np.array(s, float)
    print(f"{name:<12} | {s[:,0].mean():.3f}  | {s[:,1].mean():.4f} | {s[:,2].mean():.3f}  | {s[:,3].mean():7.1f}")
print(f"theoretical FDR for BH = pi0 * alpha = {(m - m1) / m * alpha:.4f}")
