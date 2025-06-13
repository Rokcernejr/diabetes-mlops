import pandas as pd
import numpy as np
from scipy.stats import entropy, ks_2samp
from typing import List


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
    """Compute average KS p-value across specified columns."""
    p_values = []
    for col in columns:
        if col not in ref.columns or col not in cur.columns:
            continue
        _, p_val = ks_2samp(ref[col], cur[col])
        p_values.append(p_val)
    return float(np.mean(p_values)) if p_values else 1.0


def detect_drift(
    ref: pd.DataFrame,
    cur: pd.DataFrame,
    columns: List[str],
    method: str = "kl",
    threshold: float = 0.1,
) -> bool:
    """Return True if data drift is detected."""
    if method == "kl":
        score = compute_kl_divergence(ref, cur, columns)
        return score > threshold
    else:
        p_val = compute_ks_statistic(ref, cur, columns)
        return p_val < threshold
