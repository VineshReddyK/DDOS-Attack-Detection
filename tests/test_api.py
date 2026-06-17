"""
API integration tests — run without trained models (health/info only).
Model-dependent endpoints tested with mocked registry.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def client():
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data
    assert "version" in data


def test_info_endpoint(client):
    response = client.get("/api/v1/info")
    assert response.status_code == 200
    data = response.json()
    assert "available_models" in data
    assert "class_names" in data


def test_root_redirects(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (301, 302, 307, 308)


def test_predict_no_model_returns_404(client):
    payload = {"features": [0.1] * 10, "model_type": "rf"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 404


def test_predict_invalid_model_type(client):
    payload = {"features": [0.1] * 10, "model_type": "invalid_model"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_batch_predict_no_model_returns_404(client):
    payload = {"flows": [[0.1] * 10, [0.2] * 10], "model_type": "ann"}
    response = client.post("/api/v1/predict/batch", json=payload)
    assert response.status_code == 404


def test_predict_with_mocked_rf(client):
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0])
    mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])

    from api import model_registry
    original = model_registry.registry.rf
    model_registry.registry.rf = mock_model
    model_registry.registry.class_names = ["BENIGN", "DDoS"]

    try:
        payload = {"features": [0.1] * 10, "model_type": "rf"}
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == "BENIGN"
        assert data["is_attack"] is False
        assert data["model_used"] == "rf"
    finally:
        model_registry.registry.rf = original


def test_batch_predict_with_mocked_rf(client):
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0, 1, 1])

    from api import model_registry
    original = model_registry.registry.rf
    model_registry.registry.rf = mock_model
    model_registry.registry.class_names = ["BENIGN", "DDoS"]

    try:
        payload = {"flows": [[0.1] * 10, [0.5] * 10, [0.9] * 10], "model_type": "rf"}
        response = client.post("/api/v1/predict/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert data["attack_count"] == 2
        assert abs(data["attack_rate"] - 0.6667) < 0.001
    finally:
        model_registry.registry.rf = original
