
import numpy as np
import pandas as pd
from typing import Tuple

def calibrate_garch(prices: pd.Series, p:int=1,o:int=0,q:int=1) -> Tuple[float, float, object]:
    from arch import arch_model
    log_rets = np.log(prices).diff().dropna()
    am = arch_model(log_rets*100, p=p, o=o, q=q, mean='constant', vol='GARCH', dist='normal')
    res = am.fit(disp='off')
    mu = res.params['mu']/100.0
    omega = res.params['omega']; alpha = res.params.get('alpha[1]', 0.0); beta = res.params.get('beta[1]', 0.0)
    sigma = np.sqrt(omega / (1 - alpha - beta)) / 100.0 if (1 - alpha - beta) > 1e-6 else log_rets.std()
    return mu, sigma, res

def simulate_garch_paths(S0: float, res: object, dt: float, horizon: int, n_paths: int, seed: int = 42) -> np.ndarray:
    from arch import arch_model
    rng = np.random.default_rng(seed)
    params = res.params
    mu = params['mu']/100.0
    omega = params['omega']; alpha = params.get('alpha[1]',0.0); beta = params.get('beta[1]',0.0)
    paths = np.zeros((horizon+1, n_paths))
    paths[0,:] = S0
    for j in range(n_paths):
        sigma2 = omega/(1-alpha-beta)
        eps_prev = 0.0
        for t in range(1, horizon+1):
            z = rng.standard_normal()
            eps = np.sqrt(sigma2/10000.0) * z  # convert to decimal
            r = mu*dt + eps*np.sqrt(dt)
            paths[t,j] = paths[t-1,j] * np.exp(r)
            sigma2 = omega + alpha*(eps_prev*100)**2 + beta*sigma2
            eps_prev = eps
    return paths
