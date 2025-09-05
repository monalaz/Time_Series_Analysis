
import numpy as np
import pandas as pd
from typing import Tuple

def calibrate_jd(prices: pd.Series):
    log_rets = np.log(prices).diff().dropna()
    mu = log_rets.mean()
    sigma = log_rets.std(ddof=1)
    thr = 3*sigma
    jumps = log_rets[np.abs(log_rets - mu) > thr]
    lam = len(jumps) / len(log_rets) if len(log_rets) > 0 else 0.0
    if len(jumps) > 0:
        J = jumps.values
        jm = J.mean()
        js = J.std(ddof=1) if len(J)>1 else np.abs(J[0]) if len(J)==1 else 0.0
    else:
        jm = 0.0; js = 0.0
    return mu, sigma, lam, jm, js

def simulate_merton_jd(S0: float, mu: float, sigma: float, lam: float, jmu: float, jsigma: float, dt: float, horizon: int, n_paths: int, seed:int=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    paths = np.zeros((horizon+1, n_paths))
    paths[0,:] = S0
    for t in range(1, horizon+1):
        Z = rng.standard_normal(n_paths)
        N = rng.poisson(lam*dt, size=n_paths)
        J = rng.normal(jmu, jsigma, size=n_paths) * N
        drift = (mu - 0.5*sigma**2 - lam*(np.exp(jmu + 0.5*jsigma**2)-1))*dt
        diffusion = sigma*np.sqrt(dt)*Z
        paths[t,:] = paths[t-1,:]*np.exp(drift + diffusion + J)
    return paths
