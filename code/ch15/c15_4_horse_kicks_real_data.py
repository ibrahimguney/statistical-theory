# Theory of Statistics — Chapter 15: Real Data: Deaths by Horse Kick in the Prussian Army

import numpy as np
from scipy import stats

# Bortkiewicz (1898): soldiers killed by horse kicks in 14 Prussian army corps, 1875--1894.
# Source: R package vcd, data set VonBort. Rows are corps, columns are years.
corps = [
    'G', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XIV', 'XV',
]
deaths = np.array([
    [0, 2, 2, 1, 0, 0, 1, 1, 0, 3, 0, 2, 1, 0, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 2, 0, 3, 0, 2, 0, 0, 0, 1, 1, 1, 0, 2, 0, 3, 1, 0],
    [0, 0, 0, 2, 0, 2, 0, 0, 1, 1, 0, 0, 2, 1, 1, 0, 0, 2, 0, 0],
    [0, 0, 0, 1, 1, 1, 2, 0, 2, 0, 0, 0, 1, 0, 1, 2, 1, 0, 0, 0],
    [0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 2, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 0, 2, 0, 0, 1, 2, 0, 1, 1, 3, 1, 1, 1, 0, 3, 0, 0],
    [1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 2, 0, 0, 2, 1, 0, 2, 0],
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 2, 1, 1, 1, 0, 2, 1, 1, 0, 1, 2, 0, 1, 0, 0],
    [0, 0, 1, 1, 0, 1, 0, 2, 0, 2, 0, 0, 0, 0, 2, 1, 3, 0, 1, 1],
    [0, 0, 0, 0, 2, 4, 0, 1, 3, 0, 1, 1, 1, 1, 2, 1, 3, 1, 3, 1],
    [1, 1, 2, 1, 1, 3, 0, 4, 0, 1, 0, 3, 2, 1, 0, 2, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 2, 2, 0, 0, 0, 0],
])
# the 10 corps used by Fisher (G, I, VI and XI, which were organised differently, are left out)
fisher = np.array([
    False, False, True, True, True, True, False, True, True, True, True, False, True, True,
])

def fit(x):
    lam = x.mean()                                         # Poisson MLE
    observed = np.array([np.sum(x == 0), np.sum(x == 1), np.sum(x == 2), np.sum(x >= 3)])
    p = stats.poisson.pmf([0, 1, 2], lam)
    expected = x.size * np.append(p, 1 - p.sum())
    X2 = np.sum((observed - expected)**2 / expected)
    G2 = 2 * np.sum(observed * np.log(observed / expected))
    return lam, observed, expected, X2, G2

for name, M in [("Fisher's 10 corps", deaths[fisher]), ("All 14 corps", deaths)]:
    lam, g, b, X2, G2 = fit(M.ravel())
    print(f"{name}: {M.size} corps-years, {M.sum()} deaths, lambda^ = {lam:.3f}")
    print(f"  observed (0, 1, 2, 3+) = {g} | expected = {np.round(b, 2)}")
    print(f"  fit: X^2 = {X2:.3f}, G^2 = {G2:.3f}, df = 2, p = {stats.chi2.sf(G2, 2):.3f}")
    totals = M.sum(axis=1)                                 # 20-year total per corps
    e = M.shape[1] * lam
    Gh = 2 * np.sum(totals * np.log(totals / e))           # common rate, or a separate rate per corps?
    print(f"  corps totals = {totals}")
    print(f"  homogeneity LRT: G^2 = {Gh:.2f}, df = {len(totals) - 1}, p = {stats.chi2.sf(Gh, len(totals) - 1):.4f}")
