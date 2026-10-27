# Theory of Statistics — Chapter 19: Real Data: The Efron--Morris Baseball Data

import numpy as np

# Efron and Morris (1975): hits in the first 45 at-bats of the 1970 season for 18 players, and their
# batting average over the rest of the season. Source: R package pscl, data set EfronMorris.
player = [
    'Roberto Clemente', 'Frank Robinson', 'Frank Howard', 'Jay Johnstone', 'Ken Berry',
    'Jim Spencer', 'Don Kessinger', 'Luis Alvarado', 'Ron Santo', 'Ron Swoboda', 'Del Unser',
    'Billy Williams', 'George Scott', 'Rico Petrocelli', 'Ellie Rodriguez', 'Bert Campaneris',
    'Thurman Munson', 'Max Alvis',
]
hits = np.array([
    18, 17, 16, 15, 14, 14, 13, 12, 11, 11, 10, 10, 10, 10, 10, 9, 8, 7,
])
p_season = np.array([
    0.346, 0.298, 0.276, 0.222, 0.273, 0.27, 0.263, 0.21, 0.269, 0.23, 0.264, 0.256, 0.303,
    0.264, 0.226, 0.285, 0.316, 0.2,
])
k, m = len(hits), 45
y = hits / m                                               # MLE: average over the first 45 at-bats
ybar = y.mean()
sigma2 = ybar * (1 - ybar) / m                             # binomial variance (common, approximate)
S = np.sum((y - ybar)**2)
B = (k - 3) * sigma2 / S                                   # Exercise 19.5: unbiased estimate of B
em = ybar + (1 - B) * (y - ybar)                           # Efron--Morris: shrink towards the mean
print(f"ybar = {ybar:.4f} | B^ = {B:.3f} | shrinkage factor 1 - B^ = {1 - B:.3f}")
print(f"{'player':17s}  first 45  Efron-Morris  rest of season")
for i in range(k):
    print(f"{player[i]:17s}  {y[i]:.3f}      {em[i]:.3f}          {p_season[i]:.3f}")
err = lambda t: np.sum((t - p_season)**2)
print(f"total squared error: MLE = {err(y):.4f} | Efron-Morris = {err(em):.4f} | ybar for everyone = {err(np.full(k, ybar)):.4f}")
print(f"players predicted better by Efron-Morris: {np.sum(np.abs(em - p_season) < np.abs(y - p_season))} / {k}")
