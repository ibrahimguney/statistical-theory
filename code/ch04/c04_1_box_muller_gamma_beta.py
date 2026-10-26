# Theory of Statistics — Chapter 4: Box--Muller and Gamma--Beta Independence

import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
N = 500_000
U1, U2 = rng.random(N), rng.random(N)
Z1 = np.sqrt(-2*np.log(U1)) * np.cos(2*np.pi*U2)
Z2 = np.sqrt(-2*np.log(U1)) * np.sin(2*np.pi*U2)
print(f"Box-Muller: mean=({Z1.mean():+.4f}, {Z2.mean():+.4f}) | var=({Z1.var():.4f}, {Z2.var():.4f})"
      f" | Corr={np.corrcoef(Z1, Z2)[0,1]:+.4f} | Corr(Z1^2,Z2^2)={np.corrcoef(Z1**2, Z2**2)[0,1]:+.4f}")
print(f"KS p-value (Z1 ~ N(0,1)): {stats.kstest(Z1, 'norm').pvalue:.3f}")

a, b = 2.0, 5.0
X, Y = rng.gamma(a, 1, N), rng.gamma(b, 1, N)
U, V = X + Y, X / (X + Y)
print(f"Gamma-Beta: Corr(U, V)={np.corrcoef(U, V)[0,1]:+.4f} | Corr(U, V^2)={np.corrcoef(U, V**2)[0,1]:+.4f}")
print(f"  E[V]={V.mean():.4f} (Beta mean {a/(a+b):.4f}) | E[U]={U.mean():.4f} (Gamma mean {a+b})")
print(f"  KS: V~Beta({a},{b}) p={stats.kstest(V, 'beta', args=(a, b)).pvalue:.3f}")
# A stronger check of independence: is the mean of V constant across slices of U?
q = np.quantile(U, [0, .25, .5, .75, 1])
print("  E[V] within slices of U:", np.round([V[(U >= q[i]) & (U < q[i+1])].mean() for i in range(4)], 4))
