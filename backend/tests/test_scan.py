# -*- coding: utf-8 -*-
"""
tests/test_scan.py
==================
End-to-end API tests against the **real** models: request → validation →
server-side feature extraction → DeFiGuard → explanation →
response, plus /stats accounting and input validation.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from inference_engine.api.dependencies import get_app_state
from inference_engine.api.main import app
from inference_engine.core.config import settings

TRANSFER = {
    "from": "0x3b2a1f88e4c5d6a7890bcdef2345678901bcde22",
    "to": "0x8c5d7e12f3a4b5c6d7890abcdef3456789012cd3",
    "value_eth": 0.45,
    "gas_price_gwei": 22,
    "gas_limit": 21000,
    "block_number": 19000000,
}


@pytest.fixture
def client(dg_model) -> TestClient:
    state = get_app_state()
    state.defiguard_model = dg_model
    state.defiguard_loaded = True
    state.total_predictions = state.total_fraud = 0
    state.sum_latency_ms = 0.0
    return TestClient(app)


class TestScan:
    def test_scan_returns_features_and_prediction(self, client):
        resp = client.post("/scan", json={**TRANSFER, "explain": False})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        features, pred = data["features"], data["prediction"]
        assert data["explanation"] is None
        # Features come from the server-side extractor.
        assert features["value"] == pytest.approx(0.45e18)
        assert features["effective_gas_price"] == pytest.approx(22e9)
        assert features["gas_used"] == pytest.approx(21000 * 0.7)
        assert features["length_to"] == 42.0
        assert features["gas_price_ratio"] == 1.0  # pending tx: as in training
        assert features["gas_efficiency"] == 0.0  # no cumulative block gas before inclusion
        assert 0.0 <= pred["fraud_probability"] <= 1.0
        assert pred["model"] == "defiguard"

    def test_threshold_applied(self, client, monkeypatch):
        monkeypatch.setattr(settings, "fraud_threshold", 0.3)
        pred = client.post("/scan", json=TRANSFER).json()["prediction"]
        assert pred["threshold"] == 0.3
        assert pred["is_fraud"] == (pred["fraud_probability"] > 0.3)

    def test_defiguard_scores_dataset_like_fraud_as_fraud(self, client, monkeypatch):
        """Regression: address nodes without token context collapsed DeFiGuard to
        ~0.014 for every scan. A transfer shaped like real fraud rows in the
        training data must still be flagged through /scan."""
        fraud_like = {**TRANSFER, "value_eth": 9, "gas_price_gwei": 60, "gas_limit": 30000, "block_number": 4228383}
        # (shaped like a labelled fraud row in the DeFiTransLyzer data: 9 ETH transfer, 60 gwei, block ~4.2M)
        pred = client.post("/scan", json=fraud_like).json()["prediction"]
        assert pred["fraud_probability"] > 0.8
        assert pred["is_fraud"] is True

    def test_address_case_does_not_change_score(self, client):
        upper = {**TRANSFER, "from": TRANSFER["from"].upper().replace("0X", "0x")}
        a = client.post("/scan", json=TRANSFER).json()["prediction"]
        b = client.post("/scan", json=upper).json()["prediction"]
        assert a["fraud_probability"] == b["fraud_probability"]

    def test_scan_with_explanation_is_consistent_with_verdict(self, client):
        data = client.post("/scan", json={**TRANSFER, "use_llm": True}).json()
        expl, pred = data["explanation"], data["prediction"]
        assert expl["is_fallback"] is True  # LLM disabled in tests
        assert expl["reasons"]
        if pred["is_fraud"]:
            assert expl["risk_level"] == "high"
        else:
            assert expl["risk_level"] in ("low", "medium")

    def test_each_scan_counted_once(self, client):
        client.post("/scan", json=TRANSFER)
        client.post("/scan", json={**TRANSFER, "explain": False})
        assert client.get("/stats").json()["total_predictions"] == 2

    @pytest.mark.parametrize(
        "patch",
        [
            {"value_eth": -1},
            {"gas_price_gwei": 0},
            {"gas_limit": 0},
            {"gas_limit": 21000.5},
            {"log_count": -2},
            {"to": "0x" + "a" * 200},
            {"value_eth": "abc"},
        ],
        ids=str,
    )
    def test_invalid_input_rejected(self, client, patch):
        resp = client.post("/scan", json={**TRANSFER, **patch})
        assert resp.status_code == 422

    def test_nan_rejected(self, client):
        body = '{"from": "0xa", "to": "0xb", "gas_price_gwei": NaN}'
        resp = client.post("/scan", content=body, headers={"Content-Type": "application/json"})
        assert resp.status_code == 422

    def test_scan_503_without_model(self, client):
        get_app_state().defiguard_loaded = False
        assert client.post("/scan", json=TRANSFER).status_code == 503


class TestPredictEndpointsWithRealModel:
    def test_predict_rejects_infinity(self, client):
        resp = client.post(
            "/predict", content='{"value": Infinity}', headers={"Content-Type": "application/json"}
        )
        assert resp.status_code == 422

    @pytest.mark.parametrize("patch", [{"gas_efficiency": -0.1}, {"normalized_token_transfer": 1.5}], ids=str)
    def test_predict_rejects_out_of_range(self, client, patch):
        assert client.post("/predict", json=patch).status_code == 422

    def test_predict_explain_includes_prediction_and_counts_once(self, client):
        resp = client.post("/predict/explain", json={"features": {"length_to": 42, "gas_used": 21000}})
        assert resp.status_code == 200
        data = resp.json()
        assert set(data["prediction"]) >= {"is_fraud", "fraud_probability", "threshold"}
        assert client.get("/stats").json()["total_predictions"] == 1

    def test_health_reports_model_and_threshold(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["fraud_threshold"] == 0.8


def test_scan_use_llm_flag_controls_llm(client, monkeypatch):
    """Auto-scans (use_llm false) must never reach the LLM, even with a key configured."""
    import httpx

    from inference_engine.services import explainer

    calls = []
    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    monkeypatch.setattr(
        explainer,
        "_groq_transport",
        httpx.MockTransport(lambda req: calls.append(req) or httpx.Response(
            200, json={"choices": [{"message": {"content": '{"reasons": ["r"], "recommendation": "x"}'}}]}
        )),
    )
    rules = client.post("/scan", json=TRANSFER).json()["explanation"]
    llm = client.post("/scan", json={**TRANSFER, "use_llm": True}).json()["explanation"]
    assert rules["is_fallback"] is True
    assert llm["is_fallback"] is False
    assert len(calls) == 1
