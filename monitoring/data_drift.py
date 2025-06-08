import pandas as pd
import numpy as np
from scipy.stats import entropy, ks_2samp
from typing import List


def compute_kl_divergence(ref: pd.DataFrame, cur: pd.DataFrame, columns: List[str]) -> float:
    """Compute average KL divergence across specified columns."""
    divergences = []
    for col in columns:
        ref_values = ref[col].dropna()
        cur_values = cur[col].dropna()
        combined = pd.concat([ref_values, cur_values])
        bins = np.histogram_bin_edges(combined, bins="auto")
        ref_hist, _ = np.histogram(ref_values, bins=bins, density=True)
        cur_hist, _ = np.histogram(cur_values, bins=bins, density=True)
        ref_hist += 1e-8
        cur_hist += 1e-8
        divergences.append(entropy(ref_hist, cur_hist))
    return float(np.mean(divergences))


def compute_ks_statistic(ref: pd.DataFrame, cur: pd.DataFrame, columns: List[str]) -> float:
    """Compute average Kolmogorov–Smirnov statistic across columns."""
    stats = []
    for col in columns:
        stat, _ = ks_2samp(ref[col].dropna(), cur[col].dropna())
        stats.append(stat)
    return float(np.mean(stats))


def detect_drift(
    ref: pd.DataFrame,
    cur: pd.DataFrame,
    columns: List[str] | None = None,
    method: str = "kl",
    threshold: float = 0.1,
) -> bool:
    """Return True if average divergence exceeds the given threshold."""
    if columns is None:
        columns = [c for c in ref.columns if c in cur.columns]
    if method == "kl":
        score = compute_kl_divergence(ref, cur, columns)
    elif method == "ks":
        score = compute_ks_statistic(ref, cur, columns)
    else:
        raise ValueError(f"Unsupported drift detection method: {method}")
    return score > threshold
