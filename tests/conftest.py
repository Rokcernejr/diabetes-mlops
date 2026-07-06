import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# verify_token skips auth in development; tests exercise endpoints directly
os.environ.setdefault("ENVIRONMENT", "development")

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client():
    """TestClient backed by the DummyModel.

    The model is injected straight into the ModelLoader singleton so the
    MLflow fallback chain (and its heavyweight import graph) never runs
    during unit tests.
    """
    import app.deps as deps
    from ml.dummy_model import DummyModel

    deps.ModelLoader._instance = None
    deps.ModelLoader._model = DummyModel()

    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture()
def prediction_payload():
    return {
        "race": "Caucasian",
        "gender": "Female",
        "age": "[60-70)",
        "time_in_hospital": 7,
        "num_medications": 15,
        "number_outpatient": 0,
        "number_emergency": 1,
        "number_inpatient": 0,
        "number_diagnoses": 9,
        "a1c_result": ">7",
        "max_glu_serum": "None",
        "change": "Ch",
        "diabetesMed": "Yes",
    }
