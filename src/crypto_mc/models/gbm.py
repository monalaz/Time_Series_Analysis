
import numpy as np
from typing import Tuple
import pandas as pd
from ..utils.returns import to_log_returns, ewma_vol

def calibrate_gbm(prices: pd.Series, use_ewma_vol: bool = True, ewma_lambda: float = 0.94) -> Tuple[float, float]:
    log_rets = to_log_returns(prices)
    mu = log_rets.mean()
    if use_ewma_vol:
        sigma = ewma_vol(log_rets, lam=ewma_lambda)
    else:
        sigma = log_rets.std(ddof=1)
    return mu, sigma

def simulate_gbm_paths(S0: float, mu: float, sigma: float, dt: float, horizon: int, n_paths: int, antithetic: bool = True, sobol: bool = True, seed: int = 42) -> np.ndarray:
    """Return array shape (horizon+1, n_paths)"""
    rng = np.random.default_rng(seed)
    if sobol:
        try:
            from scipy.stats.qmc import Sobol
            sampler = Sobol(d=horizon, scramble=True, seed=seed)
            n = n_paths//2 if antithetic else n_paths
            U = sampler.random(n)
            from scipy.stats import norm
            Z = norm.ppf(U.clip(1e-9, 1-1e-9))
            if antithetic:
                Z = np.vstack([Z, -Z])
        except Exception:
            Z = rng.standard_normal((n_paths, horizon))
    else:
        Z = rng.standard_normal((n_paths, horizon))
        if antithetic:
            half = n_paths//2
            Z[:half] = rng.standard_normal((half, horizon))
            Z[half:] = -Z[:half]
    paths = np.zeros((horizon+1, n_paths))
    paths[0, :] = S0
    for t in range(1, horizon+1):
        paths[t, :] = paths[t-1, :] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[:, t-1])
    return paths
