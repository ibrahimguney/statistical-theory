# Theory of Statistics — Chapter 6: Checking the Berry--Esseen Bound by Exact Computation

import numpy as np
from scipy import stats, integrate

rho = integrate.quad(lambda x: abs(x - 1)**3 * np.exp(-x), 0, np.inf)[0]   # E|X-1|^3
x = np.linspace(-6, 6, 20_001)
print(f"rho = E|X-mu|^3 = {rho:.4f}")
for n in [1, 10, 100, 1000]:
    Fn = stats.gamma.cdf(n + x*np.sqrt(n), a=n)
    sup = np.max(np.abs(Fn - stats.norm.cdf(x)))
    print(f"n={n:5d} | sup|F_n - Phi| = {sup:.5f} | Berry-Esseen bound = {0.4748*rho/np.sqrt(n):.5f} | sqrt(n)*sup = {np.sqrt(n)*sup:.4f}")
