def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert {"status", "model_status", "version", "environment"} <= data.keys()


def test_predict_probability(client, prediction_payload):
    response = client.post("/predict", json=prediction_payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["probability"] <= 1.0
