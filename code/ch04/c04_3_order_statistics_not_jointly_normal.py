# Theory of Statistics — Chapter 4: Order Statistics and a Vector That Is Not Jointly Normal

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
n, Rr = 9, 200_000
U = np.sort(rng.random((Rr, n)), axis=1)
for k in [1, 3, 5, 9]:
    print(f"U_({k}) : mean={U[:, k-1].mean():.4f} (k/(n+1)={k/(n+1):.4f}) | KS p, Beta({k},{n-k+1}): "
          f"{stats.kstest(U[:, k-1], 'beta', args=(k, n-k+1)).pvalue:.3f}")
R_ = U[:, -1] - U[:, 0]
print(f"Range: mean={R_.mean():.4f} (Beta({n-1},2) mean {(n-1)/(n+1):.4f})")

X = rng.normal(size=4000)
S = rng.choice([-1, 1], size=X.size)
Y = S * X
print(f"Y=SX: Corr(X,Y)={np.corrcoef(X, Y)[0,1]:+.4f} | Corr(X^2,Y^2)={np.corrcoef(X**2, Y**2)[0,1]:+.4f} | P(X+Y=0)={np.mean(X + Y == 0):.3f}")

rho = 0.6
Z = rng.normal(size=(4000, 2))
B = np.c_[Z[:, 0], rho*Z[:, 0] + np.sqrt(1 - rho**2)*Z[:, 1]]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True, sharey=True)
ax[0].scatter(B[:, 0], B[:, 1], s=2, alpha=0.3)
xx = np.linspace(-3.5, 3.5, 2)
ax[0].plot(xx, rho*xx, "r-", lw=2, label=r"$E[Y|X=x]=\rho x$")
ax[0].plot(xx, xx, "k:", lw=1, label="y = x")
ax[0].set_title(r"Jointly normal, $\rho=0.6$"); ax[0].legend(fontsize=8)
ax[1].scatter(X, Y, s=2, alpha=0.3, color="tab:orange")
ax[1].set_title("Normal marginals, not jointly normal (Y=SX)")
for a in ax: a.set_xlim(-3.5, 3.5); a.set_ylim(-3.5, 3.5); a.set_xlabel("x")
ax[0].set_ylabel("y"); fig.tight_layout(); fig.savefig("ortak_normal.pdf")
