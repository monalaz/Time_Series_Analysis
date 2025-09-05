
import numpy as np
import pandas as pd
from typing import Dict, List

def value_at_risk(pnl: np.ndarray, level: float=0.95) -> float:
    return -np.quantile(pnl, 1-level)

def conditional_var(pnl: np.ndarray, level: float=0.95) -> float:
    q = np.quantile(pnl, 1-level)
    tail = pnl[pnl<=q]
    return -tail.mean() if len(tail)>0 else 0.0

def max_drawdown(paths: np.ndarray) -> float:
    T, N = paths.shape[0]-1, paths.shape[1]
    mdds = []
    for j in range(N):
        s = paths[:,j]
        running_max = np.maximum.accumulate(s)
        drawdown = (s - running_max) / running_max
        mdds.append(drawdown.min())
    return float(np.mean(mdds))

def ruin_probability(paths: np.ndarray, threshold: float) -> float:
    N = paths.shape[1]
    ruined = 0
    for j in range(N):
        if (paths[:,j] <= threshold).any():
            ruined += 1
    return ruined / N
