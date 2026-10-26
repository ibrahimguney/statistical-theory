# Theory of Statistics — Chapter 23: The EM Algorithm: Genetic Linkage Data and a Normal Mixture

import numpy as np
from scipy import stats, optimize
import matplotlib.pyplot as plt

# (a) The Dempster--Laird--Rubin genetic linkage example: y = (125, 18, 20, 34),
# cell probabilities (1/2 + t/4, (1-t)/4, (1-t)/4, t/4). The first cell is split into (1/2) and (t/4).
y1, y2, y3, y4 = 125, 18, 20, 34
t, trace = 0.5, []
for _ in range(10):
    x2 = y1 * (t / 4) / (0.5 + t / 4)            # E-step: expected count falling in the t/4 part of the first cell
    t = (x2 + y4) / (x2 + y2 + y3 + y4)          # M-step: complete-data MLE of the binomial proportion
    trace.append(t)
ll = lambda t: y1*np.log(0.5 + t/4) + (y2 + y3)*np.log((1 - t)/4) + y4*np.log(t/4)
t_hat = optimize.brentq(lambda t: y1/(2 + t) - (y2 + y3)/(1 - t) + y4/t, 0.01, 0.99)
err = np.abs(np.array(trace) - t_hat)
print(f"Linkage data: EM iterations = {np.round(trace[:6], 6)}")
print(f"Direct MLE = {t_hat:.6f} | successive error ratios = {np.round(err[1:6] / err[:5], 4)}")

# (b) Two-component normal mixture
rng = np.random.default_rng(42)
n = 400
z = rng.random(n) < 0.35
x = np.where(z, rng.normal(4.0, 1.0, n), rng.normal(0.0, 1.5, n))
w, m1, m2, s1, s2 = 0.5, -1.0, 1.0, 1.0, 1.0            # deliberately poor start
def loglik(w, m1, m2, s1, s2):
    return np.log((1 - w) * stats.norm.pdf(x, m1, s1) + w * stats.norm.pdf(x, m2, s2)).sum()
hist = [loglik(w, m1, m2, s1, s2)]
for it in range(200):
    a = w * stats.norm.pdf(x, m2, s2); b = (1 - w) * stats.norm.pdf(x, m1, s1)
    g = a / (a + b)                                        # E-step: posterior membership probabilities
    w = g.mean()                                           # M-step: weighted MLEs
    m1, m2 = np.sum((1 - g) * x) / np.sum(1 - g), np.sum(g * x) / np.sum(g)
    s1 = np.sqrt(np.sum((1 - g) * (x - m1)**2) / np.sum(1 - g))
    s2 = np.sqrt(np.sum(g * (x - m2)**2) / np.sum(g))
    hist.append(loglik(w, m1, m2, s1, s2))
    if hist[-1] - hist[-2] < 1e-10: break
diffs = np.diff(hist)
print(f"Mixture: {it + 1} iterations | log-likelihood {hist[0]:.2f} -> {hist[-1]:.4f}")
print(f"Smallest increase = {diffs.min():.2e} (non-negative: monotonicity)")
print(f"Estimates: w = {w:.3f} (0.35), mu = ({m1:.3f}, {m2:.3f}) (0, 4), sigma = ({s1:.3f}, {s2:.3f}) (1.5, 1)")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ax[0].plot(hist, ".-", ms=3); ax[0].set_xlabel("iteration"); ax[0].set_ylabel(r"$\ell(\theta^{(t)})$")
ax[0].set_title("EM: the log-likelihood increases at every step", fontsize=10)
grd = np.linspace(x.min() - 1, x.max() + 1, 400)
ax[1].hist(x, bins=40, density=True, alpha=0.4)
ax[1].plot(grd, (1 - w) * stats.norm.pdf(grd, m1, s1) + w * stats.norm.pdf(grd, m2, s2), "k", lw=2)
ax[1].set_title("Fitted mixture density", fontsize=10)
fig.tight_layout(); fig.savefig("em_karisim.pdf")
