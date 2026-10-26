# Theory of Statistics — Chapter 11: Gamma MLE: Newton--Raphson

import numpy as np
from scipy.special import digamma, polygamma
from scipy import stats

rng = np.random.default_rng(42)
x = rng.gamma(2.5, 1/0.625, size=400)      # true: alpha=2.5, beta(rate)=0.625; numpy takes scale=1/rate
s = np.log(x.mean()) - np.mean(np.log(x))
a = x.mean()**2 / x.var()                         # MM starting value
print(f"MM start: alpha={a:.5f}")
for k in range(6):
    a -= (np.log(a) - digamma(a) - s) / (1/a - polygamma(1, a))
    print(f"iter {k+1}: alpha={a:.8f}")
b = a / x.mean()                                  # beta_hat = alpha_hat / xbar (rate)
a_sp, _, scale_sp = stats.gamma.fit(x, floc=0)    # scipy also returns the scale
print(f"Newton MLE : alpha={a:.5f}, beta(rate)={b:.5f}")
print(f"scipy.fit  : alpha={a_sp:.5f}, beta(rate)={1/scale_sp:.5f}")
