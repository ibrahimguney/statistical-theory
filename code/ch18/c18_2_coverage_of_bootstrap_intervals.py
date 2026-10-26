# Theory of Statistics — Chapter 18: Coverage of Bootstrap Confidence Intervals

import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
n, B, M, mu = 20, 999, 2000, 1.0
a = 0.05; zl, zu = stats.norm.ppf(a/2), stats.norm.ppf(1 - a/2)
tc = stats.t.ppf(1 - a/2, n - 1)
methods = ["t (classic)", "Percentile", "Basic", "Bootstrap-t", "BCa"]
covers = {k: 0 for k in methods}; left = {k: 0 for k in methods}; length = {k: 0.0 for k in methods}
for _ in range(M):
    x = rng.exponential(mu, n)
    xb, s = x.mean(), x.std(ddof=1); se = s/np.sqrt(n)
    xs = x[rng.integers(0, n, (B, n))]
    m_s, se_s = xs.mean(1), xs.std(1, ddof=1)/np.sqrt(n)
    ql, qu = np.percentile(m_s, [2.5, 97.5])
    t_s = (m_s - xb)/se_s
    tl, tu = np.percentile(t_s, [2.5, 97.5])
    z0 = stats.norm.ppf(np.mean(m_s < xb))
    jack = (x.sum() - x)/(n - 1); d = jack.mean() - jack
    acc = np.sum(d**3)/(6*np.sum(d**2)**1.5)
    a1 = stats.norm.cdf(z0 + (z0 + zl)/(1 - acc*(z0 + zl)))
    a2 = stats.norm.cdf(z0 + (z0 + zu)/(1 - acc*(z0 + zu)))
    iv = {"t (classic)": (xb - tc*se, xb + tc*se), "Percentile": (ql, qu), "Basic": (2*xb - qu, 2*xb - ql),
          "Bootstrap-t": (xb - tu*se, xb - tl*se), "BCa": tuple(np.percentile(m_s, [100*a1, 100*a2]))}
    for k, (L, U) in iv.items():
        covers[k] += (L <= mu <= U); left[k] += (U < mu); length[k] += U - L
print(f"{'Method':12s} | coverage | interval left of mu | mean length")
for k in methods:
    print(f"{k:12s} |  {covers[k]/M:.4f}  |       {left[k]/M:.4f}        |   {length[k]/M:.4f}")
