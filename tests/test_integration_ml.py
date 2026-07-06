"""End-to-end contract test: raw CSV -> preprocess -> train -> the persisted
artifact scores a PredictionRequest-shaped frame.

This is the test that pins the train/serve contract: if the model artifact
ever stops accepting exactly what the API sends, this fails.
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from app.schemas import PredictionRequest
from ml.predict_batch import run_batch_prediction
from ml.preprocess import preprocess_diabetes_data
from ml.train import train_diabetes_model

FAST_LGBM_PARAMS = {
    "n_estimators": 50,
    "num_leaves": 7,
    "min_child_samples": 5,
}


def _synthetic_raw_frame(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "encounter_id": np.arange(n),
            "patient_nbr": np.arange(n),
            "race": rng.choice(["Caucasian", "AfricanAmerican", "Hispanic", "?"], n),
            "gender": rng.choice(["Male", "Female"], n),
            "age": rng.choice(["[50-60)", "[60-70)", "[70-80)"], n),
            "time_in_hospital": rng.integers(1, 15, n),
            "num_medications": rng.integers(1, 41, n),
            "number_outpatient": rng.integers(0, 5, n),
            "number_emergency": rng.integers(0, 3, n),
            "number_inpatient": rng.integers(0, 3, n),
            "number_diagnoses": rng.integers(1, 10, n),
            "max_glu_serum": rng.choice(["None", ">200", ">300", "Norm"], n),
            "A1Cresult": rng.choice(["None", ">7", ">8", "Norm"], n),
            "change": rng.choice(["No", "Ch"], n),
            "diabetesMed": rng.choice(["Yes", "No"], n),
            "diag_1": rng.choice(["250.01", "401", "428", "V45"], n),
            "diag_2": rng.choice(["250.02", "276", "599", "?"], n),
            "diag_3": rng.choice(["414", "250", "710", "?"], n),
            "readmitted": rng.choice(["NO", "<30", ">30"], n),
        }
    )


@pytest.mark.integration
def test_train_serve_contract(tmp_path):
    csv_path = tmp_path / "raw.csv"
    _synthetic_raw_frame().to_csv(csv_path, index=False)

    parquet_path = tmp_path / "processed.parquet"
    df, feature_cols = preprocess_diabetes_data(csv_path, parquet_path)
    assert "readmitted" in df.columns

    model_path = tmp_path / "model.joblib"
    pipeline, metrics = train_diabetes_model(
        parquet_path,
        model_output_path=model_path,
        use_mlflow=False,
        model_params=FAST_LGBM_PARAMS,
        do_cv=False,
    )
    assert 0.0 <= metrics["auc"] <= 1.0

    # The persisted artifact must accept a PredictionRequest-shaped frame
    artifact = joblib.load(model_path)
    request = PredictionRequest(
        race="Caucasian",
        gender="Female",
        age="[60-70)",
        time_in_hospital=7,
        num_medications=15,
        number_outpatient=0,
        number_emergency=1,
        number_inpatient=0,
        number_diagnoses=9,
        a1c_result=">7",
        max_glu_serum="None",
        change="Ch",
        diabetesMed="Yes",
    )
    frame = request.as_dataframe()

    prediction = artifact.predict(frame)[0]
    probability = artifact.predict_proba(frame)[0][1]
    assert prediction in (0, 1)
    assert 0.0 <= probability <= 1.0

    # Unseen categories must encode (handle_unknown="ignore"), not crash
    unseen = frame.assign(race="Martian", a1c_result="mystery")
    assert 0.0 <= artifact.predict_proba(unseen)[0][1] <= 1.0


@pytest.mark.integration
def test_batch_prediction_uses_same_artifact(tmp_path):
    csv_path = tmp_path / "raw.csv"
    _synthetic_raw_frame().to_csv(csv_path, index=False)

    parquet_path = tmp_path / "processed.parquet"
    preprocess_diabetes_data(csv_path, parquet_path)

    model_path = tmp_path / "model.joblib"
    train_diabetes_model(
        parquet_path,
        model_output_path=model_path,
        use_mlflow=False,
        model_params=FAST_LGBM_PARAMS,
        do_cv=False,
    )
    artifact = joblib.load(model_path)

    batch_input = pd.DataFrame(
        {
            "race": ["Caucasian", "Hispanic"],
            "gender": ["Female", "Male"],
            "age": [65, "[50-60)"],
            "time_in_hospital": [7, 3],
            "num_medications": [15, 5],
            "number_outpatient": [0, 1],
            "number_emergency": [1, 0],
            "number_inpatient": [0, 0],
            "number_diagnoses": [9, 4],
            "a1c_result": [">7", "None"],
            "max_glu_serum": ["None", "Norm"],
            "change": ["Ch", "No"],
            "diabetesMed": ["Yes", "No"],
        }
    )

    result = run_batch_prediction(artifact, batch_input)
    assert len(result) == 2
    assert result["probability"].between(0.0, 1.0).all()


@pytest.mark.integration
def test_shap_explains_real_pipeline(tmp_path):
    shap = pytest.importorskip("shap")  # noqa: F841

    from app.shap_utils import explain_prediction as shap_explain

    csv_path = tmp_path / "raw.csv"
    _synthetic_raw_frame().to_csv(csv_path, index=False)
    parquet_path = tmp_path / "processed.parquet"
    preprocess_diabetes_data(csv_path, parquet_path)

    pipeline, _ = train_diabetes_model(
        parquet_path,
        model_output_path=None,
        use_mlflow=False,
        model_params=FAST_LGBM_PARAMS,
        do_cv=False,
    )

    frame = PredictionRequest(
        race="Caucasian",
        gender="Female",
        age="[60-70)",
        time_in_hospital=7,
        num_medications=15,
        number_outpatient=0,
        number_emergency=1,
        number_inpatient=0,
        number_diagnoses=9,
        a1c_result=">7",
        max_glu_serum="None",
        change="Ch",
        diabetesMed="Yes",
    ).as_dataframe()

    feature_names, shap_values, base_value = shap_explain(pipeline, frame)
    assert feature_names, "expected SHAP to explain the tree model inside the pipeline"
    assert len(feature_names) == len(shap_values)
    assert isinstance(base_value, float)
