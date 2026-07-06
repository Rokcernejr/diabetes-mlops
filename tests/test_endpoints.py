"""Regression tests for the endpoint bugs found in the July 2026 review:
GET-with-body explain, no-op model reload, duplicate metric registration,
and missing 422/503 coverage.
"""

import sys
import types


def test_predict_invalid_payload_returns_422(client, prediction_payload):
    bad = dict(prediction_payload, time_in_hospital=99)
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_without_model_returns_503(client, prediction_payload):
    from app.main import app

    saved = app.state.model
    app.state.model = None
    try:
        assert client.post("/predict", json=prediction_payload).status_code == 503
        assert client.get("/ready").status_code == 503
        assert client.get("/health").json()["status"] == "degraded"
    finally:
        app.state.model = saved


def test_explain_is_a_post_endpoint(client, prediction_payload):
    response = client.post("/predict/explain", json=prediction_payload)
    assert response.status_code == 200
    data = response.json()
    assert {"feature_names", "shap_values", "base_value", "top_features"} <= set(
        data.keys()
    )

    # GET with a required body was the original bug; the route is POST-only now
    assert client.get("/predict/explain").status_code == 405


def test_explain_returns_501_without_shap(client, prediction_payload, monkeypatch):
    # An empty module makes `from app.shap_utils import ...` raise ImportError,
    # simulating an environment without shap installed
    monkeypatch.setitem(
        sys.modules, "app.shap_utils", types.ModuleType("app.shap_utils")
    )
    response = client.post("/predict/explain", json=prediction_payload)
    assert response.status_code == 501


def test_model_reload_returns_a_new_model_object(client, monkeypatch):
    import app.deps as deps
    from app.main import app
    from ml.dummy_model import DummyModel

    monkeypatch.setattr(
        deps.ModelLoader, "_load_fresh", lambda self: DummyModel(), raising=True
    )

    before = app.state.model
    response = client.post("/model/reload")
    assert response.status_code == 200

    # The original bug: the singleton returned its cache, so "reload" was a no-op
    assert app.state.model is not before


def test_model_reload_requires_auth_outside_development(client, monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    response = client.post("/model/reload")
    assert response.status_code == 401


def test_metrics_module_coexists_with_main(client):
    # Importing app.metrics alongside app.main used to crash with
    # "Duplicated timeseries in CollectorRegistry"
    import app.metrics  # noqa: F401

    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"api_requests_total" in response.content
