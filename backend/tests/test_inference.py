# -*- coding: utf-8 -*-
"""
tests/test_inference.py
=========================
Schema validation, /health, /predict, /predict/explain and /extract with a
mocked DeFiGuard model (the real-model path is covered in test_scan.py).
"""
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from inference_engine.api.dependencies import get_app_state
from inference_engine.api.main import app
from inference_engine.core.config import settings
from models.features import ALL_FEATURES

TX: Dict[str, Any] = {
    "from": "0x742d35cc6634c0532925a3b844bc454e4438f44e",
    "to": "0x9f44b3d4a44f91c9e7dc7b8e72f44f929d3c3bdd",
    "value": hex(10**18), "gas": hex(21000), "gasPrice": hex(20 * 10**9), "nonce": "0x3", "input": "0x",
}
RECEIPT: Dict[str, Any] = {"from": TX["from"], "to": TX["to"], "gasUsed": hex(21000), "status": "0x1", "logs": []}


@pytest.fixture()
def mock_dg_model() -> MagicMock:
    """A minimal mock of DeFiGuardPredictor (pending threshold 0.6, receipt 0.4)."""
    model = MagicMock()
    model.thresholds = {"pending": 0.6, "receipt": 0.4}
    model.score.side_effect = lambda tx, receipt=None: {
        "fraud_probability": 0.85, "mode": "receipt" if receipt else "pending"}
    return model


def _returns(model: MagicMock, prob: float) -> None:
    model.score.side_effect = lambda tx, receipt=None: {
        "fraud_probability": prob, "mode": "receipt" if receipt else "pending"}


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


class TestHealthEndpoint:
    def test_health_ok_reports_per_mode_thresholds(self, app_client: TestClient) -> None:
        data = app_client.get("/health").json()
        assert data == {
            "status": "ok",
            "model_loaded": True,
            "model_path": "/mock/defi_guard_model.joblib",
            "thresholds": {"pending": 0.6, "receipt": 0.4},
            "llm_configured": False,
            "llm_model": "openai/gpt-oss-120b",
        }

    def test_health_reports_threshold_override(self, app_client, monkeypatch) -> None:
        monkeypatch.setattr(settings, "fraud_threshold", 0.7)
        assert app_client.get("/health").json()["thresholds"] == {"pending": 0.7, "receipt": 0.7}

    def test_health_degraded_when_model_not_loaded(self, unloaded_client: TestClient) -> None:
        resp = unloaded_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "degraded"
        assert resp.json()["thresholds"] == {}


class TestPredictEndpoint:
    def test_predict_response_contract(self, app_client: TestClient) -> None:
        data = app_client.post("/predict", json={"tx": TX}).json()
        assert set(data) == {"is_fraud", "fraud_probability", "threshold", "mode", "model", "exec_time_ms"}
        assert data["model"] == "defiguard"
        assert data["mode"] == "pending"
        assert data["threshold"] == 0.6

    def test_receipt_selects_receipt_mode_and_threshold(self, app_client, mock_dg_model) -> None:
        data = app_client.post("/predict", json={"tx": TX, "receipt": RECEIPT}).json()
        assert data["mode"] == "receipt"
        assert data["threshold"] == 0.4
        tx, receipt = mock_dg_model.score.call_args.args
        assert tx == TX and receipt == RECEIPT

    def test_fraud_at_or_above_threshold(self, app_client, mock_dg_model) -> None:
        assert app_client.post("/predict", json={"tx": TX}).json()["is_fraud"] is True
        _returns(mock_dg_model, 0.6)
        assert app_client.post("/predict", json={"tx": TX}).json()["is_fraud"] is True

    def test_legitimate_below_threshold(self, app_client, mock_dg_model) -> None:
        _returns(mock_dg_model, 0.05)
        assert app_client.post("/predict", json={"tx": TX}).json()["is_fraud"] is False

    def test_threshold_override(self, app_client, mock_dg_model, monkeypatch) -> None:
        monkeypatch.setattr(settings, "fraud_threshold", 0.9)
        data = app_client.post("/predict", json={"tx": TX}).json()
        assert data["threshold"] == 0.9
        assert data["is_fraud"] is False

    def test_missing_tx_rejected(self, app_client: TestClient) -> None:
        assert app_client.post("/predict", json={}).status_code == 422

    def test_malformed_payload_is_422_not_500(self, app_client, mock_dg_model) -> None:
        mock_dg_model.score.side_effect = TypeError("logs must be a list")
        assert app_client.post("/predict", json={"tx": TX}).status_code == 422

    @pytest.mark.parametrize("path", ["/predict", "/predict/explain", "/scan"])
    def test_503_when_model_not_loaded(self, unloaded_client: TestClient, path: str) -> None:
        body = {"gas_price_gwei": 20} if path == "/scan" else {"tx": TX}
        assert unloaded_client.post(path, json=body).status_code == 503

    def test_non_finite_model_output_is_an_error(self, mock_dg_model) -> None:
        _returns(mock_dg_model, float("nan"))
        state = get_app_state()
        state.defiguard_model, state.defiguard_loaded = mock_dg_model, True
        client = TestClient(app, raise_server_exceptions=False)
        assert client.post("/predict", json={"tx": TX}).status_code == 500


class TestPredictExplainEndpoint:
    def test_explain_high_risk_transaction(self, app_client: TestClient) -> None:
        data = app_client.post("/predict/explain", json={"tx": TX, "use_llm": False}).json()
        assert data["risk_level"] == "high"
        assert data["prediction"]["is_fraud"] is True
        assert data["reasons"]
        assert data["recommendation"]

    def test_explain_low_risk_transaction(self, app_client, mock_dg_model) -> None:
        _returns(mock_dg_model, 0.05)
        data = app_client.post("/predict/explain", json={"tx": TX}).json()
        assert data["risk_level"] == "low"
        assert data["reasons"]

    def test_pii_address_redaction(self) -> None:
        from inference_engine.services.explainer import mask_address
        addr = "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
        assert mask_address(addr) == "0x742d...f44e"


class TestExtractEndpoint:
    def test_pending_payload(self, app_client: TestClient) -> None:
        data = app_client.post("/extract", json={"tx_payload": TX}).json()
        assert data["mode"] == "pending"
        assert list(data["features"]) == ALL_FEATURES
        assert data["features"]["gas_used_ratio"] == 0.0

    def test_tx_and_receipt_payload(self, app_client: TestClient) -> None:
        data = app_client.post("/extract", json={"tx_payload": {"tx": TX, "receipt": RECEIPT}}).json()
        assert data["mode"] == "receipt"
        assert data["features"]["gas_used_ratio"] == 1.0

    def test_malformed_payload_rejected(self, app_client: TestClient) -> None:
        resp = app_client.post("/extract", json={"tx_payload": {"tx": "not-an-object"}})
        assert resp.status_code == 422
