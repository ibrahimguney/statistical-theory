# Theory of Statistics — Chapter 19: Prior, Likelihood and Posterior; Bernstein--von Mises

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

t = np.linspace(0, 0.5, 600)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
prior = stats.beta(2, 18)
ax[0].plot(t, prior.pdf(t), "--", label="prior Beta(2,18)")
ax[0].plot(t, stats.beta(11, 41).pdf(t), ":", label="likelihood (normalised)")
ax[0].plot(t, stats.beta(12, 58).pdf(t), "k", lw=2, label="posterior Beta(12,58)")
ax[0].set_xlabel(r"$\theta$"); ax[0].legend(fontsize=8); ax[0].set_title("n = 50, x = 10")

rng = np.random.default_rng(42)
theta0 = 0.2
for n in [20, 200, 2000]:
    x = rng.binomial(n, theta0)
    ps = stats.beta(2 + x, 18 + n - x)
    mle = x / n
    bvm = stats.norm(mle, np.sqrt(mle*(1-mle)/n))
    tt = np.linspace(0, 0.5, 2000)
    tv = 0.5*np.sum(np.abs(ps.pdf(tt) - bvm.pdf(tt))) * (tt[1] - tt[0])
    ax[1].plot(tt, ps.pdf(tt), label=f"n={n}")
    print(f"n={n:5d} | posterior mean={ps.mean():.4f} | MLE={mle:.4f} | TV(posterior, BvM normal)={tv:.4f}")
ax[1].axvline(theta0, color="k", ls=":"); ax[1].set_xlim(0, 0.45); ax[1].legend(fontsize=8)
ax[1].set_xlabel(r"$\theta$"); ax[1].set_title("Concentration of the posterior")
fig.tight_layout(); fig.savefig("bayes_sonsal.pdf")
