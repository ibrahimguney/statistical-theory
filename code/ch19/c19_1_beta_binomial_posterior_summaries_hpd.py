# Theory of Statistics — Chapter 19: Beta--Binomial: Posterior Summaries and the HPD Interval

import scipy.stats as stats
import numpy as np
from scipy.optimize import minimize_scalar

a_post, b_post = 2 + 10, 18 + 40
print(f"Bayes estimate: {a_post / (a_post + b_post):.4f}")
print(f"Credible interval: {stats.beta.ppf([0.025, 0.975], a_post, b_post)}")

post = stats.beta(a_post, b_post)
print(f"Posterior median = {post.median():.4f} | mode (MAP) = {(a_post-1)/(a_post+b_post-2):.4f} | sd = {post.std():.4f}")
# HPD: the u minimising the length of [ppf(u), ppf(u+0.95)]
w = lambda u: post.ppf(u + 0.95) - post.ppf(u)
u = minimize_scalar(w, bounds=(0, 0.05), method="bounded").x
L, U = post.ppf(u), post.ppf(u + 0.95)
print(f"95% HPD = [{L:.4f}, {U:.4f}] (length {U-L:.4f}) | density at endpoints: {post.pdf(L):.3f}, {post.pdf(U):.3f}")
lo, hi = post.ppf([0.025, 0.975])
print(f"Equal-tailed length = {hi-lo:.4f}")
print(f"P(theta > 0.15 | x) = {post.sf(0.15):.4f}")
