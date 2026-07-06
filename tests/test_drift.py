import numpy as np
import pandas as pd

from monitoring.data_drift import (
    compute_kl_divergence,
    compute_ks_statistic,
    detect_drift,
)


def test_kl_divergence_zero():
    # Seeded: with unseeded samples this fails by construction some % of runs
    rng = np.random.RandomState(42)
    df1 = pd.DataFrame({"a": rng.normal(0, 1, 1000)})
    df2 = pd.DataFrame({"a": rng.normal(0, 1, 1000)})
    kl = compute_kl_divergence(df1, df2, ["a"])
    assert kl < 1.0


def test_ks_statistic_zero():
    # Seeded: the p > 0.1 assertion rejects ~10% of unseeded identical samples
    rng = np.random.RandomState(42)
    df1 = pd.DataFrame({"a": rng.normal(0, 1, 1000)})
    df2 = pd.DataFrame({"a": rng.normal(0, 1, 1000)})
    p_val = compute_ks_statistic(df1, df2, ["a"])
    # With identical distributions we expect a high p-value. Using
    # a 10% significance level provides 90% confidence that no drift
    # exists between samples.
    assert p_val > 0.1


def test_detect_drift_threshold():
    np.random.seed(0)
    ref = pd.DataFrame({"a": np.random.normal(0, 1, 1000)})
    cur = pd.DataFrame({"a": np.random.normal(5, 1, 1000)})
    # Detect drift when the KS test rejects the null at 10% significance
    assert detect_drift(ref, cur, ["a"], method="ks", threshold=0.1)
    assert detect_drift(ref, cur, ["a"], method="kl", threshold=0.02)
