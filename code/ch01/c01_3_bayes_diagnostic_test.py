# Theory of Statistics — Chapter 1: Bayes' Formula: Simulating a Diagnostic Test

import numpy as np

rng = np.random.default_rng(42)
N = 2_000_000
ill = rng.random(N) < 0.01
positive = np.where(ill, rng.random(N) < 0.99, rng.random(N) < 0.05)
print(f"P(D | +) theory    : {0.0099/0.0594:.4f}")
print(f"P(D | +) simulation: {ill[positive].mean():.4f}")
print(f"P(+)     theory    : {0.0594:.4f} | simulation: {positive.mean():.4f}")
