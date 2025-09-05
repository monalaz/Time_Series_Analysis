
import numpy as np
import pandas as pd

def moving_block_bootstrap(log_rets: pd.Series, horizon: int, n_paths: int, block_size:int=5, seed:int=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = log_rets.values
    n = len(r)
    paths = np.zeros((horizon, n_paths))
    for j in range(n_paths):
        t = 0
        while t < horizon:
            start = rng.integers(0, max(1, n - block_size))
            block = r[start:start+block_size]
            k = min(block_size, horizon - t)
            paths[t:t+k, j] = block[:k]
            t += k
    return paths  # returns, not prices
