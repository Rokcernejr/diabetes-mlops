import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from ml.train import validate_data, split_data


def _make_df(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feat1": np.arange(n),
            "feat2": np.random.randint(0, 5, size=n),
            "readmitted": [0] * (n // 2) + [1] * (n - n // 2),
        }
    )


def test_validate_data_pass():
    df = _make_df(50)
    validate_data(df)


def test_validate_data_missing_target():
    df = pd.DataFrame({"feat": [1, 2, 3]})
    with pytest.raises(ValueError):
        validate_data(df, target_col="readmitted")


def test_split_data_stratified():
    df = _make_df(100)
    X_train, X_test, y_train, y_test = split_data(
        df, target_col="readmitted", test_size=0.25, random_state=0
    )
    assert len(X_train) == 75
    assert len(X_test) == 25
    # Stratification keeps roughly equal class balance
    assert abs(y_train.mean() - 0.5) < 0.05
    assert abs(y_test.mean() - 0.5) < 0.05

