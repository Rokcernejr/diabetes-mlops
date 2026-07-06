"""Single source of truth for the train/serve feature contract.

The API request schema (app.schemas.PredictionRequest) and the trained model
must always agree on features. The model is trained on exactly the columns a
PredictionRequest supplies (API naming), and all preprocessing lives inside
the persisted sklearn Pipeline — so any frame the pipeline could score at
training time it can also score at serving time.
"""

from __future__ import annotations

import pandas as pd

# Dataset columns -> API field names (everything else matches 1:1)
DATASET_TO_API = {"A1Cresult": "a1c_result"}

CATEGORICAL_FEATURES = [
    "race",
    "gender",
    "age",
    "a1c_result",
    "max_glu_serum",
    "change",
    "diabetesMed",
]

NUMERIC_FEATURES = [
    "time_in_hospital",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

SERVING_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

DEFAULT_LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "random_state": 42,
    "n_estimators": 1000,  # reduced by early stopping
}


def dataset_to_serving_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select and rename the serving features from a (preprocessed) dataset frame."""
    df = df.rename(columns=DATASET_TO_API)
    missing = [c for c in SERVING_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing serving features: {missing}")
    X = df[SERVING_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)
    return X


def build_model_pipeline(model_params: dict | None = None):
    """Build the preprocessing + LightGBM pipeline persisted as one artifact.

    OneHotEncoder(handle_unknown="ignore") guarantees unseen categories at
    inference time encode to all-zeros instead of crashing.
    """
    from lightgbm import LGBMClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    params = dict(DEFAULT_LGBM_PARAMS)
    if model_params:
        params.update(model_params)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )

    return Pipeline(
        [
            ("prep", preprocessor),
            ("clf", LGBMClassifier(**params)),
        ]
    )
