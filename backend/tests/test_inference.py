# -*- coding: utf-8 -*-
"""
tests/test_inference.py
=========================
Schema validation, /health, /predict and /predict/explain with a mocked
DeFiGuard model (the real-model path is covered in test_scan.py).
"""
from __future__ import annotations

from typing import Dict
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from inference_engine.api.dependencies import get_app_state
from inference_engine.api.main import app


@pytest.fixture()
def mock_dg_model() -> MagicMock:
    """A minimal mock of DeFiGuardPredictor."""
    model = MagicMock()
    model.predict_fraud_proba.return_value = 0.85
    return model


@pytest.fixture()
def app_client(mock_dg_model: MagicMock) -> TestClient:
    """TestClient with the mocked model loaded in the shared state."""
    state = get_app_state()
    state.defiguard_model = mock_dg_model
    state.defiguard_model_path = "/mock/defi_guard_model.joblib"
    state.defiguard_loaded = True
    return TestClient(app)


@pytest.fixture()
def unloaded_client() -> TestClient:
    state = get_app_state()
    state.defiguard_model = None
    state.defiguard_loaded = False
    state.defiguard_model_path = ""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def minimal_features() -> Dict[str, float]:
    return {
        "length_transaction_hash": 66.0,
        "length_to": 42.0,
        "length_from": 42.0,
        "block_number": 19_500_000.0,
        "gas_used": 21_000.0,
        "chain_id": 1.0,
        "total_gas_cost": 4_200_000_000_000.0,
        "effective_gas_price": 200_000_000_000.0,
        "value": 1e18,
    }


class TestTransactionFeatures:
    def test_valid_minimal_payload_parses(self, minimal_features: Dict[str, float]) -> None:
        from inference_engine.api.schemas import TransactionFeatures
        assert TransactionFeatures(**minimal_features).chain_id == 1.0

    def test_defaults_match_pending_transaction_training_values(self) -> None:
        from inference_engine.api.schemas import TransactionFeatures
        feat = TransactionFeatures()
        assert feat.gas_used == 0.0
        assert feat.gas_price_ratio == 1.0
        assert feat.event_activity_flag == 0.0

    def test_out_of_range_is_same_address_rejected(self) -> None:
        from pydantic import ValidationError
        from inference_engine.api.schemas import TransactionFeatures
        with pytest.raises(ValidationError):
            TransactionFeatures(is_same_address=2.0)


class TestHealthEndpoint:
    def test_health_ok_when_model_loaded(self, app_client: TestClient) -> None:
        data = app_client.get("/health").json()
        assert data == {
            "status": "ok",
            "model_loaded": True,
            "model_path": "/mock/defi_guard_model.joblib",
            "fraud_threshold": 0.8,
            "llm_configured": False,
            "llm_model": "openai/gpt-oss-120b",
        }

    def test_health_degraded_when_model_not_loaded(self, unloaded_client: TestClient) -> None:
        resp = unloaded_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "degraded"


class TestPredictEndpoint:
    def test_predict_response_contract(self, app_client: TestClient, minimal_features) -> None:
        data = app_client.post("/predict", json=minimal_features).json()
        assert set(data) == {"is_fraud", "fraud_probability", "threshold", "model", "exec_time_ms"}
        assert data["model"] == "defiguard"
        assert data["threshold"] == 0.8

    def test_fraud_above_threshold(self, app_client: TestClient, minimal_features) -> None:
        data = app_client.post("/predict", json=minimal_features).json()
        assert data["is_fraud"] is True
        assert data["fraud_probability"] == 0.85

    def test_legitimate_below_threshold(self, app_client, minimal_features, mock_dg_model) -> None:
        mock_dg_model.predict_fraud_proba.return_value = 0.05
        assert app_client.post("/predict", json=minimal_features).json()["is_fraud"] is False

    def test_threshold_is_exclusive(self, app_client, minimal_features, mock_dg_model) -> None:
        mock_dg_model.predict_fraud_proba.return_value = 0.8
        assert app_client.post("/predict", json=minimal_features).json()["is_fraud"] is False

    def test_feature_vector_reaches_model_without_address_graph(self, app_client, mock_dg_model) -> None:
        app_client.post("/predict", json={"value": 5.0})
        features, raw_tx = mock_dg_model.predict_fraud_proba.call_args.args
        assert features["value"] == 5.0
        assert raw_tx is None

    def test_empty_payload_uses_defaults(self, app_client: TestClient) -> None:
        assert app_client.post("/predict", json={}).status_code == 200

    @pytest.mark.parametrize("path", ["/predict", "/predict/explain", "/scan"])
    def test_503_when_model_not_loaded(self, unloaded_client: TestClient, path: str) -> None:
        body = {"gas_price_gwei": 20} if path == "/scan" else {}
        assert unloaded_client.post(path, json=body).status_code == 503

    def test_non_finite_model_output_is_an_error(self, mock_dg_model) -> None:
        mock_dg_model.predict_fraud_proba.return_value = float("nan")
        state = get_app_state()
        state.defiguard_model, state.defiguard_loaded = mock_dg_model, True
        client = TestClient(app, raise_server_exceptions=False)
        assert client.post("/predict", json={}).status_code == 500


class TestPredictExplainEndpoint:
    def test_explain_high_risk_transaction(self, app_client: TestClient, minimal_features) -> None:
        payload = {
            "features": minimal_features,
            "raw_from": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
            "raw_to": "0x9f44b3d4a44f91c9e7dc7b8e72f44f929d3c3bdd",
            "amount_eth": 12.5,
        }
        data = app_client.post("/predict/explain", json=payload).json()
        assert data["risk_level"] == "high"
        assert data["prediction"]["is_fraud"] is True
        assert data["reasons"]
        assert data["recommendation"]

    def test_explain_low_risk_transaction(self, app_client, minimal_features, mock_dg_model) -> None:
        mock_dg_model.predict_fraud_proba.return_value = 0.05
        low = {**minimal_features, "value": 1e17, "effective_gas_price": 20e9}
        data = app_client.post("/predict/explain", json={"features": low}).json()
        assert data["risk_level"] == "low"
        assert data["reasons"]

    def test_pii_address_redaction(self) -> None:
        from inference_engine.services.explainer import mask_address
        addr = "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
        assert mask_address(addr) == "0x742d...f44e"
