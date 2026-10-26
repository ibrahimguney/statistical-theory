# Theory of Statistics — Chapter 11: Analytic and Numerical MLE in the Normal Model; Invariance

import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(42)
data = rng.normal(10.0, 3.0, size=250)
mu_an, var_an = np.mean(data), np.var(data, ddof=0)
print(f"Analytic MLE: mu={mu_an:.4f}, s2={var_an:.4f}, sd={np.sqrt(var_an):.4f}")

def negll(par, param):
    mu, s = par
    sd = {"sd": abs(s), "var": np.sqrt(abs(s)), "logsd": np.exp(s)}[param]
    return -np.sum(-np.log(sd) - 0.5*np.log(2*np.pi) - (data - mu)**2 / (2*sd**2))

for param, x0 in [("sd", [5, 1]), ("var", [5, 1]), ("logsd", [5, 1])]:
    r = minimize(negll, x0, args=(param,), method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10})
    mu, s = r.x
    sd = {"sd": abs(s), "var": np.sqrt(abs(s)), "logsd": np.exp(s)}[param]
    print(f"Numerical ({param:5s} param.): mu={mu:.4f}, sd={sd:.4f}, -logL={r.fun:.4f}")
