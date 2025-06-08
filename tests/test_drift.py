import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np
from monitoring.data_drift import compute_kl_divergence, compute_ks_statistic, detect_drift



def test_kl_divergence_zero():
    df1 = pd.DataFrame({'a': [1, 2, 3, 4]})
    df2 = pd.DataFrame({'a': [1, 2, 3, 4]})
    kl = compute_kl_divergence(df1, df2, ['a'])
    assert kl == 0.0


def test_ks_statistic_zero():
    df1 = pd.DataFrame({'a': [0, 1, 2, 3]})
    df2 = pd.DataFrame({'a': [0, 1, 2, 3]})
    ks = compute_ks_statistic(df1, df2, ['a'])
    assert ks == 0.0


def test_detect_drift_threshold():
    np.random.seed(0)
    ref = pd.DataFrame({'a': np.random.normal(0, 1, 1000)})
    cur = pd.DataFrame({'a': np.random.normal(5, 1, 1000)})
    assert detect_drift(ref, cur, ['a'], method='ks', threshold=0.1)
    assert detect_drift(ref, cur, ['a'], method='kl', threshold=0.1)
