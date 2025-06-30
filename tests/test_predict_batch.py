import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd

from app.schemas import PredictionRequest
from ml.predict_batch import age_to_range, dataframe_to_payloads


def test_age_to_range_numeric():
    assert age_to_range(45) == "[40-50)"
    assert age_to_range("25") == "[20-30)"
    # already formatted strings are returned unchanged
    assert age_to_range("[60-70)") == "[60-70)"


def test_payload_generation():
    df = pd.DataFrame(
        {
            "race": ["Caucasian"],
            "gender": ["Female"],
            "age": [45],
            "time_in_hospital": [5],
            "num_medications": [10],
            "number_outpatient": [0],
            "number_emergency": [0],
            "number_inpatient": [0],
            "number_diagnoses": [6],
            "a1c_result": [">7"],
            "max_glu_serum": ["None"],
            "change": ["Ch"],
            "diabetesMed": ["Yes"],
            "extra": [1],
        }
    )

    payloads = dataframe_to_payloads(df)
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["age"] == "[40-50)"
    assert set(payload.keys()) == set(PredictionRequest.model_fields.keys())
