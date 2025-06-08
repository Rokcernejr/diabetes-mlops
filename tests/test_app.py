import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client(monkeypatch):
    """Create TestClient with patched model loading."""
    import app.deps as deps

    def _raise(*args, **kwargs):
        raise RuntimeError("mlflow disabled in tests")

    monkeypatch.setattr(deps.mlflow.pyfunc, "load_model", _raise)

    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert {"status", "model_status", "version", "environment"} <= data.keys()


def test_predict_probability(client):
    payload = {
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
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["probability"] <= 1.0
