# -*- coding: utf-8 -*-
"""
tests/test_explain.py
======================
Tests for POST /predict/explain and the underlying explainer service.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import math

import pytest

from feature_extractor.extractor import extract_features
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

SENDER = "0x742d35cc6634c0532925a3b844bc9e7595f44e2a"
SAFE_TX = {"from": SENDER, "to": "0x8c5d7e12f3a4b5c6d7890abcdef3456789012cd3", "value": "0x0",
           "gas": hex(21000), "gasPrice": hex(20 * 10**9), "nonce": "0x10", "input": "0x"}
# Self-addressed token approval at 200 gwei moving 15 ETH from a fresh account.
FRAUD_TX = {"from": SENDER, "to": SENDER, "value": hex(15 * 10**18), "gas": hex(400_000),
            "gasPrice": hex(200 * 10**9), "nonce": "0x0", "input": "0x095ea7b3" + "00" * 64}
SAFE_FEATURES = extract_features(SAFE_TX)
FRAUD_FEATURES = extract_features(FRAUD_TX)
THRESHOLD = 0.8


def test_rule_fallback_low_risk():
    result = _build_rule_fallback_explanation(SAFE_FEATURES, False, 0.1, THRESHOLD)
    assert result["risk_level"] == "low"
    assert len(result["reasons"]) >= 1
    assert "safe" in result["recommendation"].lower()
    assert result["is_fallback"] is True


def test_rule_fallback_high_risk():
    result = _build_rule_fallback_explanation(FRAUD_FEATURES, True, 0.9, THRESHOLD)
    assert result["risk_level"] == "high"
    assert len(result["reasons"]) >= 3
    assert "CRITICAL" in result["recommendation"]
    assert result["is_fallback"] is True


def test_rule_fallback_medium_risk():
    medium_features = extract_features({**SAFE_TX, "gasPrice": hex(60 * 10**9)})
    result = _build_rule_fallback_explanation(medium_features, False, 0.55, THRESHOLD)
    assert any("elevated" in r for r in result["reasons"])
    assert result["risk_level"] == "medium"
    assert result["is_fallback"] is True


def test_rule_fallback_self_transfer_masks_address():
    result = _build_rule_fallback_explanation(
        FRAUD_FEATURES,
        True,
        0.9,
        THRESHOLD,
        raw_from="0x742d35Cc6634C0532925a3b844Bc9e7595f44e2a",
    )
    full_addr = "0x742d35Cc6634C0532925a3b844Bc9e7595f44e2a"
    reasons_text = " ".join(result["reasons"])
    assert "Self-transfer" in reasons_text
    assert full_addr not in reasons_text  # PII not exposed


def test_rule_fallback_names_drainer_signals():
    reasons = " ".join(_build_rule_fallback_explanation(FRAUD_FEATURES, True, 0.9, THRESHOLD)["reasons"])
    assert "approval" in reasons
    assert "nonce 0" in reasons
    assert "200 Gwei" in reasons


# ---------------------------------------------------------------------------
# explain_transaction_risk — fallback path (no Groq key)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_explain_transaction_risk_uses_fallback_when_no_key():
    result = await explain_transaction_risk(
        features=SAFE_FEATURES,
        is_fraud=False,
        fraud_probability=0.1,
        threshold=THRESHOLD,
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
        threshold=THRESHOLD,
    )
    assert result["risk_level"] == "high"


# ---------------------------------------------------------------------------
# FastAPI endpoint handler — explain_risk
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_dg_model():
    model = MagicMock()
    model.score.return_value = {"fraud_probability": 0.9, "mode": "pending"}
    model.thresholds = {"pending": 0.5, "receipt": 0.5}
    return model


@pytest.mark.asyncio
async def test_explain_endpoint_returns_valid_schema(mock_dg_model):
    from inference_engine.api.main import explain_risk
    from inference_engine.api.schemas import RiskExplanationRequest

    request = RiskExplanationRequest(tx=FRAUD_TX, use_llm=False)

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
    assert result.prediction.mode == "pending"
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
    result = await explain_transaction_risk(FRAUD_FEATURES, is_fraud=True, fraud_probability=0.9, threshold=THRESHOLD,
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
    result = await explain_transaction_risk(FRAUD_FEATURES, is_fraud=True, fraud_probability=0.95, threshold=THRESHOLD)
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
    result = await explain_transaction_risk(SAFE_FEATURES, is_fraud=False, fraud_probability=0.1, threshold=THRESHOLD)
    assert result["is_fallback"] is True
    assert result["risk_level"] == "low"


@pytest.mark.asyncio
async def test_llm_call_budget_enforced(groq, monkeypatch):
    monkeypatch.setattr(settings, "llm_max_calls_per_minute", 1)
    calls = groq(_json.dumps({"reasons": ["r"], "recommendation": "x"}))
    first = await explain_transaction_risk(SAFE_FEATURES, is_fraud=False, fraud_probability=0.1, threshold=THRESHOLD)
    second = await explain_transaction_risk(SAFE_FEATURES, is_fraud=False, fraud_probability=0.1, threshold=THRESHOLD)
    assert first["is_fallback"] is False
    assert second["is_fallback"] is True
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_use_llm_false_skips_llm(groq):
    calls = groq(_json.dumps({"reasons": ["r"], "recommendation": "x"}))
    result = await explain_transaction_risk(SAFE_FEATURES, is_fraud=False, fraud_probability=0.1, threshold=THRESHOLD, use_llm=False)
    assert result["is_fallback"] is True
    assert calls == []


def test_risk_bands_follow_applied_threshold():
    assert explainer.risk_level_for(False, 0.7, 0.8) == "medium"
    assert explainer.risk_level_for(False, 0.3, 0.8) == "low"
    assert explainer.risk_level_for(True, 0.81, 0.8) == "high"
    assert explainer.risk_level_for(False, 0.39, 0.4) == "low"
    assert explainer.risk_level_for(False, 0.45, 0.4) == "medium"  # not blocked, above the floor


# ---------------------------------------------------------------------------
# Fallback reasons — an AI explanation that could not be produced says why
# ---------------------------------------------------------------------------

def _groq_error(status, message):
    return httpx.MockTransport(lambda req: httpx.Response(status, json={"error": {"message": message}}))


@pytest.mark.asyncio
async def test_no_key_reports_reason():
    result = await explain_transaction_risk(SAFE_FEATURES, is_fraud=False, fraud_probability=0.1, threshold=THRESHOLD)
    assert result["is_fallback"] is True
    assert result["fallback_reason"] == "No GROQ_API_KEY configured."


@pytest.mark.asyncio
async def test_rule_based_by_request_has_no_reason(groq):
    groq(_json.dumps({"reasons": ["r"], "recommendation": "x"}))
    result = await explain_transaction_risk(SAFE_FEATURES, is_fraud=False, fraud_probability=0.1, threshold=THRESHOLD, use_llm=False)
    assert result["fallback_reason"] is None


@pytest.mark.asyncio
async def test_retired_model_reported_clearly(monkeypatch, caplog):
    """Regression: Groq retired llama-3.3-70b-versatile; every call 404'd silently."""
    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    monkeypatch.setattr(settings, "groq_model", "llama-3.3-70b-versatile")
    monkeypatch.setattr(explainer, "_groq_transport", _groq_error(
        404, "The model `llama-3.3-70b-versatile` does not exist or you do not have access to it."))
    with caplog.at_level("ERROR"):
        result = await explain_transaction_risk(SAFE_FEATURES, is_fraud=False, fraud_probability=0.1, threshold=THRESHOLD)
    assert result["is_fallback"] is True
    assert result["fallback_reason"] == "Groq model 'llama-3.3-70b-versatile' is unavailable; set GROQ_MODEL."
    assert "set GROQ_MODEL" in caplog.text


@pytest.mark.parametrize(
    "transport,reason",
    [
        (_groq_error(500, "boom"), "Groq API error (HTTP 500)."),
        (httpx.MockTransport(lambda req: (_ for _ in ()).throw(httpx.ConnectError("down"))), "Could not reach Groq."),
        (httpx.MockTransport(lambda req: (_ for _ in ()).throw(httpx.ReadTimeout("slow"))), "Groq did not respond in time."),
    ],
    ids=["http-500", "unreachable", "timeout"],
)
@pytest.mark.asyncio
async def test_failure_reasons(monkeypatch, transport, reason):
    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    monkeypatch.setattr(explainer, "_groq_transport", transport)
    result = await explain_transaction_risk(SAFE_FEATURES, is_fraud=False, fraud_probability=0.1, threshold=THRESHOLD)
    assert result["fallback_reason"] == reason


@pytest.mark.asyncio
async def test_configured_model_is_sent(groq, monkeypatch):
    monkeypatch.setattr(settings, "groq_model", "qwen/qwen3.8-27b")
    calls = groq(_json.dumps({"reasons": ["r"], "recommendation": "x"}))
    result = await explain_transaction_risk(SAFE_FEATURES, is_fraud=False, fraud_probability=0.1, threshold=THRESHOLD)
    assert calls[0]["model"] == "qwen/qwen3.8-27b"
    assert result["fallback_reason"] is None


@pytest.mark.asyncio
async def test_prompt_reports_human_scale_values(groq):
    """Log-scaled model features are converted back before reaching the prompt."""
    calls = groq(_json.dumps({"reasons": ["r"], "recommendation": "x"}))
    features = {**FRAUD_FEATURES, "log_n_transfers": math.log1p(3)}
    await explain_transaction_risk(features, True, 0.9, THRESHOLD)
    prompt = calls[0]["messages"][1]["content"]
    assert "Token Transfers in Receipt: 3" in prompt
    assert "Gas Price: 200.0 Gwei" in prompt
    assert "Transfer Amount: 15.000000 ETH" in prompt
    assert "Token Approval Call: True" in prompt
