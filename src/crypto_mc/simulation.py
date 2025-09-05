
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from .models.gbm import calibrate_gbm, simulate_gbm_paths
from .models.garch import calibrate_garch, simulate_garch_paths
from .models.jump_diffusion import calibrate_jd, simulate_merton_jd
from .models.bootstrap import moving_block_bootstrap
from .utils.returns import to_log_returns

def simulate_single(prices: pd.Series, model: str, horizon:int, n_paths:int, dt:float=1.0, config:dict=None, seed:int=42) -> np.ndarray:
    S0 = float(prices.iloc[-1])
    if model == "gbm":
        use_ewma = config.get('models',{}).get('gbm',{}).get('use_ewma_vol', True)
        lam = config.get('models',{}).get('gbm',{}).get('ewma_lambda', 0.94)
        mu, sigma = calibrate_gbm(prices, use_ewma_vol=use_ewma, ewma_lambda=lam)
        paths = simulate_gbm_paths(S0, mu, sigma, dt, horizon, n_paths, antithetic=True, sobol=True, seed=seed)
    elif model == "garch":
        p = config.get('models',{}).get('garch',{}).get('p',1)
        o = config.get('models',{}).get('garch',{}).get('o',0)
        q = config.get('models',{}).get('garch',{}).get('q',1)
        mu, sigma, res = calibrate_garch(prices, p,o,q)
        paths = simulate_garch_paths(S0, res, dt, horizon, n_paths, seed=seed)
    elif model == "jump":
        mu, sigma, lam, jm, js = calibrate_jd(prices)
        paths = simulate_merton_jd(S0, mu, sigma, lam, jm, js, dt, horizon, n_paths, seed=seed)
    elif model == "bootstrap":
        logrets = to_log_returns(prices)
        block = config.get('bootstrap',{}).get('block_size',5)
        R = moving_block_bootstrap(logrets, horizon, n_paths, block_size=block, seed=seed)
        S = np.zeros((horizon+1, n_paths))
        S[0,:] = S0
        for t in range(1, horizon+1):
            S[t,:] = S[t-1,:]*np.exp(R[t-1,:])
        paths = S
    else:
        raise ValueError("Unknown model: "+model)
    return paths

def simulate_multi(prices_map: Dict[str, pd.Series], model: str, horizon:int, n_paths:int, dt:float=1.0, seed:int=42, config:dict=None, weights:List[float]=None):
    import numpy as np
    symbols = list(prices_map.keys())
    logrets = [np.log(prices_map[s]).diff().dropna().values for s in symbols]
    minlen = min(len(r) for r in logrets)
    R = np.column_stack([r[-minlen:] for r in logrets])
    mu_hat = R.mean(axis=0)
    cov = np.cov(R, rowvar=False)
    L = np.linalg.cholesky(cov + 1e-12*np.eye(cov.shape[0]))
    rng = np.random.default_rng(seed)
    S0 = np.array([float(prices_map[s].iloc[-1]) for s in symbols])
    paths = np.zeros((horizon+1, n_paths, len(symbols)))
    paths[0,:,:] = S0
    for t in range(1, horizon+1):
        Z = rng.standard_normal((n_paths, len(symbols)))
        dR = Z @ L.T
        drift = (mu_hat - 0.5*np.diag(cov)) * dt
        paths[t,:,:] = paths[t-1,:,:] * np.exp(drift + dR*np.sqrt(dt))
    if weights is not None:
        w = np.array(weights); w = w/np.sum(w)
        port = (paths * w.reshape(1,1,-1)).sum(axis=2)
        return paths, port
    return paths, None
