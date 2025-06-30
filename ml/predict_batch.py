import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from app.schemas import PredictionRequest

AGE_RANGES = [f"[{i}-{i+10})" for i in range(0, 100, 10)]


def age_to_range(value: Any) -> str:
    """Convert numeric age to categorical range used by the API."""
    if isinstance(value, str) and value.startswith("[") and value.endswith(")"):
        return value
    try:
        age = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid age value: {value}") from exc
    if age < 0 or age >= 100:
        raise ValueError("age must be between 0 and 99")
    lower = (age // 10) * 10
    upper = lower + 10
    return f"[{lower}-{upper})"


def load_csv(file_path: Path | str) -> pd.DataFrame:
    """Read CSV data and normalise the age column."""
    df = pd.read_csv(file_path)
    if "age" in df.columns:
        df["age"] = df["age"].apply(age_to_range)
    return df


def extract_request_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the columns required by ``PredictionRequest``."""
    required = list(PredictionRequest.model_fields.keys())
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[required].copy()


def dataframe_to_payloads(df: pd.DataFrame) -> list[dict]:
    """Convert a dataframe into JSON-ready payloads for ``/predict``."""
    pred_df = extract_request_columns(df)
    if "age" in pred_df.columns:
        pred_df["age"] = pred_df["age"].apply(age_to_range)
    payloads = []
    for _, row in pred_df.iterrows():
        req = PredictionRequest(**row.to_dict())
        payloads.append(req.model_dump())
    return payloads


def save_payloads(payloads: Iterable[dict], output: Path | str) -> None:
    """Save prediction payloads to a JSON file."""
    with open(output, "w", encoding="utf-8") as f:
        json.dump(list(payloads), f, indent=2)


def run_batch_prediction(model: Any, df: pd.DataFrame) -> pd.DataFrame:
    """Run predictions on a dataframe using a loaded model."""
    payload_df = pd.DataFrame(dataframe_to_payloads(df))
    preds = model.predict(payload_df)
    probas = model.predict_proba(payload_df)[:, 1]
    result = df.copy()
    result["prediction"] = preds
    result["probability"] = probas
    return result
