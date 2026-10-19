# -*- coding: utf-8 -*-
"""
tests/test_explain.py
======================
Tests for POST /predict/explain and the underlying explainer service.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from inference_engine.services.explainer import (
    _build_rule_fallback_explanation,
    explain_transaction_risk,
    mask_address,
)


# ---------------------------------------------------------------------------
# mask_address
# ---------------------------------------------------------------------------

def test_mask_address_normal():
    addr = "0x742d35Cc6634C0532925a3b844Bc9e7595f44e2a"
    result = mask_address(addr)
    assert result.startswith("0x742d")
    assert result.endswith("4e2a")
    assert "..." in result


def test_mask_address_short():
    assert mask_address("0x1234") == "0x1234"


def test_mask_address_none():
    assert mask_address(None) == "0x..."


def test_mask_address_empty():
    assert mask_address("") == "0x..."


# ---------------------------------------------------------------------------
# _build_rule_fallback_explanation
# ---------------------------------------------------------------------------

SAFE_FEATURES = {
    "gas_price_ratio": 1.0,
    "effective_gas_price": 20e9,
    "value": 0.0,
    "is_same_address": 0.0,
    "gas_used": 21000.0,
    "log_removed": 0.0,
}

FRAUD_FEATURES = {
    "gas_price_ratio": 4.5,
    "effective_gas_price": 200e9,
    "value": 15e18,
    "is_same_address": 1.0,
    "gas_used": 400_000.0,
    "log_removed": 1.0,
}


def test_rule_fallback_low_risk():
    result = _build_rule_fallback_explanation(SAFE_FEATURES, False, 0.1)
    assert result["risk_level"] == "low"
    assert len(result["reasons"]) >= 1
    assert "safe" in result["recommendation"].lower()
    assert result["is_fallback"] is True


def test_rule_fallback_high_risk():
    result = _build_rule_fallback_explanation(FRAUD_FEATURES, True, 0.9)
    assert result["risk_level"] == "high"
    assert len(result["reasons"]) >= 3
    assert "CRITICAL" in result["recommendation"]
    assert result["is_fallback"] is True


def test_rule_fallback_medium_risk():
    medium_features = {**SAFE_FEATURES, "effective_gas_price": 60e9}
    result = _build_rule_fallback_explanation(medium_features, False, 0.55)
    assert result["risk_level"] == "medium"
    assert result["is_fallback"] is True


def test_rule_fallback_self_transfer_masks_address():
    result = _build_rule_fallback_explanation(
        {**FRAUD_FEATURES, "is_same_address": 1.0},
        True,
        0.9,
        raw_from="0x742d35Cc6634C0532925a3b844Bc9e7595f44e2a",
    )
    full_addr = "0x742d35Cc6634C0532925a3b844Bc9e7595f44e2a"
    reasons_text = " ".join(result["reasons"])
    assert full_addr not in reasons_text  # PII not exposed


# ---------------------------------------------------------------------------
# explain_transaction_risk — fallback path (no Groq key)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_explain_transaction_risk_uses_fallback_when_no_key():
    result = await explain_transaction_risk(
        features=SAFE_FEATURES,
        is_fraud=False,
        fraud_probability=0.1,
    )
    assert "risk_level" in result
    assert "reasons" in result
    assert "recommendation" in result
    assert result["is_fallback"] is True
    assert result["exec_time_ms"] >= 0


@pytest.mark.asyncio
async def test_explain_transaction_risk_high_fraud():
    result = await explain_transaction_risk(
        features=FRAUD_FEATURES,
        is_fraud=True,
        fraud_probability=0.95,
    )
    assert result["risk_level"] == "high"


# ---------------------------------------------------------------------------
# FastAPI endpoint handler — explain_risk
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_dg_model():
    model = MagicMock()
    model.predict_fraud_proba.return_value = 0.9
    return model


@pytest.mark.asyncio
async def test_explain_endpoint_returns_valid_schema(mock_dg_model):
    from inference_engine.api.main import explain_risk
    from inference_engine.api.schemas import RiskExplanationRequest, TransactionFeatures

    request = RiskExplanationRequest(
        features=TransactionFeatures(
            length_transaction_hash=66,
            length_to=42,
            length_from=42,
            is_same_address=1.0,
            gas_efficiency=1.0,
            effective_gas_price=200e9,
        ),
        raw_from="0x742d35Cc6634C0532925a3b844Bc9e7595f44e2a",
        raw_to="0x742d35Cc6634C0532925a3b844Bc9e7595f44e2a",
    )

    result = await explain_risk(
        request=request,
        model=mock_dg_model,
    )

    assert result.risk_level in ("low", "medium", "high")
    assert isinstance(result.reasons, list)
    assert len(result.reasons) >= 1
    assert isinstance(result.recommendation, str)
    assert result.exec_time_ms >= 0
    assert result.prediction is not None
    assert result.prediction.is_fraud is True
    # The explanation's risk level must agree with the model verdict.
    assert result.risk_level == "high"


# ---------------------------------------------------------------------------
# Groq LLM boundary — exercised through httpx.MockTransport (no network)
# ---------------------------------------------------------------------------

import json as _json

import httpx

from inference_engine.core.config import settings
from inference_engine.services import explainer


def _groq_reply(content, status=200):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(_json.loads(request.content))
        return httpx.Response(status, json={"choices": [{"message": {"content": content}}]})

    return httpx.MockTransport(handler), calls


@pytest.fixture
def groq(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "test-key")

    def install(content, status=200):
        transport, calls = _groq_reply(content, status)
        monkeypatch.setattr(explainer, "_groq_transport", transport)
        return calls

    return install


@pytest.mark.asyncio
async def test_llm_explanation_used_when_valid(groq):
    calls = groq(_json.dumps({"reasons": ["Gas is 10x base fee."], "recommendation": "Do NOT sign."}))
    result = await explain_transaction_risk(FRAUD_FEATURES, is_fraud=True, fraud_probability=0.9,
                                            raw_from="0x742d35Cc6634C0532925a3b844Bc9e7595f44e2a")
    assert result["is_fallback"] is False
    assert result["reasons"] == ["Gas is 10x base fee."]
    assert result["risk_level"] == "high"
    prompt = calls[0]["messages"][1]["content"]
    assert "Verdict: BLOCKED" in prompt
    assert "0x742d35Cc6634C0532925a3b844Bc9e7595f44e2a" not in prompt  # address masked


@pytest.mark.asyncio
async def test_llm_cannot_contradict_model_verdict(groq):
    groq(_json.dumps({"risk_level": "low", "reasons": ["Looks fine."], "recommendation": "Proceed."}))
    result = await explain_transaction_risk(FRAUD_FEATURES, is_fraud=True, fraud_probability=0.95)
    assert result["risk_level"] == "high"


@pytest.mark.parametrize(
    "content,status",
    [
        ("not json at all", 200),
        (_json.dumps({"reasons": "a string, not a list", "recommendation": "x"}), 200),
        (_json.dumps({"reasons": [], "recommendation": "x"}), 200),
        (_json.dumps({"reasons": ["ok"], "recommendation": 42}), 200),
        (_json.dumps(["a", "list"]), 200),
        (_json.dumps({"reasons": ["ok"], "recommendation": "x"}), 500),
    ],
    ids=["non-json", "reasons-str", "reasons-empty", "rec-not-str", "top-level-list", "http-500"],
)
@pytest.mark.asyncio
async def test_malformed_llm_output_falls_back(groq, content, status):
    groq(content, status)
    result = await explain_transaction_risk(SAFE_FEATURES, is_fraud=False, fraud_probability=0.1)
    assert result["is_fallback"] is True
    assert result["risk_level"] == "low"


@pytest.mark.asyncio
async def test_llm_call_budget_enforced(groq, monkeypatch):
    monkeypatch.setattr(settings, "llm_max_calls_per_minute", 1)
    calls = groq(_json.dumps({"reasons": ["r"], "recommendation": "x"}))
    first = await explain_transaction_risk(SAFE_FEATURES, is_fraud=False, fraud_probability=0.1)
    second = await explain_transaction_risk(SAFE_FEATURES, is_fraud=False, fraud_probability=0.1)
    assert first["is_fallback"] is False
    assert second["is_fallback"] is True
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_use_llm_false_skips_llm(groq):
    calls = groq(_json.dumps({"reasons": ["r"], "recommendation": "x"}))
    result = await explain_transaction_risk(SAFE_FEATURES, is_fraud=False, fraud_probability=0.1, use_llm=False)
    assert result["is_fallback"] is True
    assert calls == []


def test_risk_bands_follow_configured_threshold(monkeypatch):
    monkeypatch.setattr(settings, "fraud_threshold", 0.8)
    assert explainer.risk_level_for(False, 0.7) == "medium"
    assert explainer.risk_level_for(False, 0.3) == "low"
    assert explainer.risk_level_for(True, 0.81) == "high"
    monkeypatch.setattr(settings, "fraud_threshold", 0.4)
    assert explainer.risk_level_for(False, 0.39) == "low"
    assert explainer.risk_level_for(False, 0.45) == "medium"  # not blocked, above the floor
