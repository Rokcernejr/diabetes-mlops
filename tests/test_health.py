import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

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


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")

