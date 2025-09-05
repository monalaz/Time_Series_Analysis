
import numpy as np
import pandas as pd

def to_log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices).diff().dropna()

def ewma_vol(log_rets: pd.Series, lam: float = 0.94) -> float:
    # RiskMetrics-style EWMA
    s2 = 0.0
    for r in log_rets:
        s2 = lam * s2 + (1 - lam) * r * r
    return np.sqrt(s2)
