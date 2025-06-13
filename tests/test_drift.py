import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from monitoring.data_drift import compute_kl_divergence, compute_ks_statistic, detect_drift

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def test_kl_divergence_zero():
    df1 = pd.DataFrame({"a": np.random.normal(0, 1, 1000)})
    df2 = pd.DataFrame({"a": np.random.normal(0, 1, 1000)})
    kl = compute_kl_divergence(df1, df2, ["a"])
    assert kl < 1.0


def test_ks_statistic_zero():
    df1 = pd.DataFrame({"a": np.random.normal(0, 1, 1000)})
    df2 = pd.DataFrame({"a": np.random.normal(0, 1, 1000)})
    p_val = compute_ks_statistic(df1, df2, ["a"])
    assert p_val > 0.05


def test_detect_drift_threshold():
    np.random.seed(0)
    ref = pd.DataFrame({"a": np.random.normal(0, 1, 1000)})
    cur = pd.DataFrame({"a": np.random.normal(5, 1, 1000)})
    assert detect_drift(ref, cur, ["a"], method="ks", threshold=0.05)
    assert detect_drift(ref, cur, ["a"], method="kl", threshold=0.02)

