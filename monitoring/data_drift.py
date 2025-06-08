import pandas as pd
import numpy as np
from typing import List
from scipy.stats import entropy, ks_2samp


def compute_kl_divergence(ref: pd.DataFrame, cur: pd.DataFrame, columns: List[str]) -> float:
    """Compute average KL divergence across specified columns."""
    divergences = []
    for col in columns:
        if col not in ref.columns or col not in cur.columns:
            continue
        ref_dist = np.histogram(ref[col], bins=20, density=True)[0] + 1e-9
        cur_dist = np.histogram(cur[col], bins=20, density=True)[0] + 1e-9
        divergences.append(entropy(ref_dist, cur_dist))
    return float(np.mean(divergences)) if divergences else 0.0


def compute_ks_statistic(ref: pd.DataFrame, cur: pd.DataFrame, columns: List[str]) -> float:
    """Compute average KS statistic across specified columns."""
    stats = []
    for col in columns:
        if col not in ref.columns or col not in cur.columns:
            continue
        stat, _ = ks_2samp(ref[col], cur[col])
        stats.append(stat)
    return float(np.mean(stats)) if stats else 0.0


def detect_drift(ref: pd.DataFrame, cur: pd.DataFrame, columns: List[str], method: str = "kl", threshold: float = 0.1) -> bool:
    """Return True if average divergence exceeds the threshold."""
    if method == "kl":
        score = compute_kl_divergence(ref, cur, columns)
    else:
        score = compute_ks_statistic(ref, cur, columns)
    return score > threshold
