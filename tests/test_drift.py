import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from monitoring.data_drift import compute_kl_divergence, compute_ks_statistic, detect_drift


def test_kl_divergence_zero():
    ref = pd.DataFrame({"a": np.random.normal(0, 1, 1000)})
    cur = ref.copy()
    kl = compute_kl_divergence(ref, cur, ["a"])
    assert kl < 1e-6


def test_detect_drift():
    ref = pd.DataFrame({"a": np.random.normal(0, 1, 1000)})
    cur = pd.DataFrame({"a": np.random.normal(5, 1, 1000)})
    assert detect_drift(ref, cur, ["a"], method="ks", threshold=0.1)
