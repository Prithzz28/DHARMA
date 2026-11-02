# -*- coding: utf-8 -*-
"""
tests/test_scan.py
==================
End-to-end API tests against the **real** model: request → validation →
server-side feature extraction → DeFiGuard → explanation → response, plus
/stats accounting and input validation.
"""
from __future__ import annotations

import math

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
    "nonce": 57,
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
    def test_scan_returns_features_and_prediction(self, client, dg_model):
        resp = client.post("/scan", json={**TRANSFER, "explain": False})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        features, pred = data["features"], data["prediction"]
        assert data["explanation"] is None
        # Features come from the shared extractor (log-scaled).
        assert math.expm1(features["log_value_eth"]) == pytest.approx(0.45)
        assert math.expm1(features["log_gas_price_gwei"]) == pytest.approx(22)
        assert math.expm1(features["log_nonce"]) == pytest.approx(57)
        assert features["has_calldata"] == 0.0
        assert features["gas_used_ratio"] == 0.0  # no receipt before signing
        assert pred["mode"] == "pending"
        assert pred["threshold"] == dg_model.thresholds["pending"]
        assert 0.0 <= pred["fraud_probability"] <= 1.0
        assert pred["is_fraud"] == (pred["fraud_probability"] >= pred["threshold"])
        assert pred["model"] == "defiguard"

    def test_calldata_reaches_the_model(self, client):
        plain = client.post("/scan", json=TRANSFER).json()
        call = client.post("/scan", json={**TRANSFER, "data": "0x095ea7b3" + "00" * 64}).json()
        assert call["features"]["sel_approval"] == 1.0
        assert call["prediction"]["fraud_probability"] != plain["prediction"]["fraud_probability"]

    def test_threshold_override_applied(self, client, monkeypatch):
        monkeypatch.setattr(settings, "fraud_threshold", 0.3)
        pred = client.post("/scan", json=TRANSFER).json()["prediction"]
        assert pred["threshold"] == 0.3
        assert pred["is_fraud"] == (pred["fraud_probability"] >= 0.3)

    def test_address_identity_does_not_change_score(self, client):
        other = {**TRANSFER, "from": "0x" + "1" * 40, "to": "0x" + "2" * 40}
        a = client.post("/scan", json=TRANSFER).json()["prediction"]
        b = client.post("/scan", json=other).json()["prediction"]
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
            {"nonce": -2},
            {"data": "0xzz"},
            {"data": "0xabc"},
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
    TX = {"from": TRANSFER["from"], "to": TRANSFER["to"], "value": hex(10**18), "gas": hex(21000),
          "gasPrice": hex(22 * 10**9), "nonce": "0x3", "input": "0x"}
    RECEIPT = {"from": TRANSFER["from"], "to": TRANSFER["to"], "gasUsed": hex(21000), "status": "0x1", "logs": []}

    def test_predict_receipt_mode(self, client, dg_model):
        data = client.post("/predict", json={"tx": self.TX, "receipt": self.RECEIPT}).json()
        assert data["mode"] == "receipt"
        assert data["threshold"] == dg_model.thresholds["receipt"]
        assert data["fraud_probability"] == pytest.approx(
            dg_model.score(self.TX, self.RECEIPT)["fraud_probability"], abs=1e-6)

    def test_predict_rejects_infinity(self, client):
        resp = client.post("/predict", content='{"tx": {"value": Infinity}}',
                           headers={"Content-Type": "application/json"})
        assert resp.status_code == 422

    def test_predict_explain_includes_prediction_and_counts_once(self, client):
        resp = client.post("/predict/explain", json={"tx": self.TX, "use_llm": False})
        assert resp.status_code == 200
        data = resp.json()
        assert set(data["prediction"]) >= {"is_fraud", "fraud_probability", "threshold", "mode"}
        assert client.get("/stats").json()["total_predictions"] == 1

    def test_health_reports_model_thresholds(self, client, dg_model):
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["thresholds"] == dg_model.thresholds


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
