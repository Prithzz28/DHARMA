# -*- coding: utf-8 -*-
"""
tests/test_inference.py
=========================
Unit and integration tests for the inference engine.

Tests
------
- Pydantic schema validation (valid/invalid inputs).
- /predict endpoint with a mocked Random Forest model.
- /health endpoint in loaded and unloaded states.
- Feature ordering in the prediction service.
"""
from __future__ import annotations

import math
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture()
def mock_rf_model() -> MagicMock:
    """A minimal mock of a RandomForestClassifier."""
    model = MagicMock()
    # predict_proba returns [[p_legit, p_fraud]] shape (1, 2)
    model.predict_proba.return_value = np.array([[0.15, 0.85]])
    model.n_estimators = 200
    return model


@pytest.fixture()
def app_client(mock_rf_model: MagicMock) -> TestClient:
    """FastAPI TestClient with the model pre-loaded in the dependency state."""
    # Import here so patching happens before the module loads the model.
    from inference_engine.api.dependencies import _model_state
    from inference_engine.api.main import app

    _model_state.model = mock_rf_model
    _model_state.model_path = "/mock/random_forest_fraud_model.joblib"
    _model_state.is_loaded = True

    return TestClient(app)


@pytest.fixture()
def minimal_features() -> Dict[str, float]:
    """A minimal valid feature vector (all numeric fields at zero defaults)."""
    return {
        "length_transaction_hash": 66.0,
        "length_to": 42.0,
        "length_from": 42.0,
        "block_number": 19_500_000.0,
        "gas_used": 21_000.0,
        "chain_id": 1.0,
        "total_gas_cost": 4_200_000_000_000.0,
        "effective_gas_price": 200_000_000_000.0,
        "cumulative_gas_used": 500_000.0,
        "gas_price_ratio": 2.5,
        "length_log": math.log(2),
        "value": 1e18,
    }


# --------------------------------------------------------------------------- #
# Schema tests
# --------------------------------------------------------------------------- #

class TestTransactionFeatures:
    def test_valid_minimal_payload_parses(self, minimal_features: Dict[str, float]) -> None:
        from inference_engine.api.schemas import TransactionFeatures
        feat = TransactionFeatures(**minimal_features)
        assert feat.chain_id == 1.0

    def test_defaults_fill_missing_fields(self) -> None:
        from inference_engine.api.schemas import TransactionFeatures
        feat = TransactionFeatures()  # all fields at defaults
        assert feat.gas_used == 0.0
        assert feat.is_same_address == 0.0
        assert feat.event_activity_flag == 0.0

    def test_out_of_range_is_same_address_rejected(self) -> None:
        from pydantic import ValidationError
        from inference_engine.api.schemas import TransactionFeatures
        with pytest.raises(ValidationError):
            TransactionFeatures(is_same_address=2.0)  # must be 0.0–1.0


# --------------------------------------------------------------------------- #
# /health endpoint
# --------------------------------------------------------------------------- #

class TestHealthEndpoint:
    def test_health_ok_when_model_loaded(self, app_client: TestClient) -> None:
        resp = app_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    def test_health_degraded_when_model_not_loaded(self) -> None:
        from inference_engine.api.dependencies import _model_state
        from inference_engine.api.main import app

        original_state = (_model_state.model, _model_state.is_loaded, _model_state.model_path)
        _model_state.model = None
        _model_state.is_loaded = False
        _model_state.model_path = ""

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"

        # Restore
        _model_state.model, _model_state.is_loaded, _model_state.model_path = original_state


# --------------------------------------------------------------------------- #
# /predict endpoint
# --------------------------------------------------------------------------- #

class TestPredictEndpoint:
    def test_predict_returns_fraud_verdict(
        self, app_client: TestClient, minimal_features: Dict[str, float]
    ) -> None:
        resp = app_client.post("/predict", json=minimal_features)
        assert resp.status_code == 200
        data = resp.json()
        assert "is_fraud" in data
        assert "fraud_probability" in data
        assert "exec_time_ms" in data

    def test_predict_fraud_above_threshold(
        self, app_client: TestClient, minimal_features: Dict[str, float]
    ) -> None:
        # Mock returns 0.85 probability — above default threshold of 0.65
        resp = app_client.post("/predict", json=minimal_features)
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_fraud"] is True
        assert data["fraud_probability"] > 0.65

    def test_predict_legitimate_below_threshold(
        self, app_client: TestClient, minimal_features: Dict[str, float], mock_rf_model: MagicMock
    ) -> None:
        # Override mock to return low fraud probability
        mock_rf_model.predict_proba.return_value = np.array([[0.95, 0.05]])
        resp = app_client.post("/predict", json=minimal_features)
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_fraud"] is False

    def test_predict_empty_payload_uses_defaults(self, app_client: TestClient) -> None:
        resp = app_client.post("/predict", json={})
        assert resp.status_code == 200

    def test_predict_503_when_model_not_loaded(self) -> None:
        from inference_engine.api.dependencies import _model_state
        from inference_engine.api.main import app

        original = (_model_state.model, _model_state.is_loaded)
        _model_state.model = None
        _model_state.is_loaded = False

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/predict", json={})
        assert resp.status_code == 503

        _model_state.model, _model_state.is_loaded = original


# --------------------------------------------------------------------------- #
# Prediction service unit tests
# --------------------------------------------------------------------------- #

class TestPredictionService:
    def test_feature_array_correct_shape(self) -> None:
        from inference_engine.services.prediction import _build_feature_array, FEATURE_ORDER
        features = {k: float(i) for i, k in enumerate(FEATURE_ORDER)}
        arr = _build_feature_array(features)
        assert arr.shape == (1, len(FEATURE_ORDER))
        assert arr.dtype == np.float32

    def test_missing_features_default_to_zero(self) -> None:
        from inference_engine.services.prediction import _build_feature_array
        arr = _build_feature_array({})
        assert (arr == 0.0).all()


# --------------------------------------------------------------------------- #
# /predict/explain endpoint tests
# --------------------------------------------------------------------------- #

class TestPredictExplainEndpoint:
    def test_explain_high_risk_transaction(
        self, app_client: TestClient, minimal_features: Dict[str, float]
    ) -> None:
        payload = {
            "features": minimal_features,
            "raw_from": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
            "raw_to": "0x9f44b3d4a44f91c9e7dc7b8e72f44f929d3c3bdd",
            "amount_eth": 12.5
        }
        resp = app_client.post("/predict/explain", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "risk_level" in data
        assert data["risk_level"] in ("high", "medium", "low")
        assert "reasons" in data
        assert isinstance(data["reasons"], list)
        assert len(data["reasons"]) > 0
        assert "recommendation" in data

    def test_explain_low_risk_transaction(
        self, app_client: TestClient, minimal_features: Dict[str, float], mock_rf_model: MagicMock
    ) -> None:
        mock_rf_model.predict_proba.return_value = np.array([[0.95, 0.05]])
        low_risk_features = dict(minimal_features)
        low_risk_features["gas_price_ratio"] = 1.0
        low_risk_features["value"] = 1e17

        payload = {"features": low_risk_features}
        resp = app_client.post("/predict/explain", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["risk_level"] == "low"
        assert len(data["reasons"]) > 0

    def test_pii_address_redaction(self) -> None:
        from inference_engine.services.explainer import mask_address
        addr = "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
        masked = mask_address(addr)
        assert masked == "0x742d...f44e"
        assert addr not in masked
