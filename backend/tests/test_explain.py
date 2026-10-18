# -*- coding: utf-8 -*-
"""
tests/test_explain.py
======================
Tests for POST /predict/explain and the underlying explainer service.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

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
    medium_features = {**SAFE_FEATURES, "gas_price_ratio": 2.2}
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
    with patch("inference_engine.services.explainer.settings") as mock_settings:
        mock_settings.groq_api_key = ""
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
    with patch("inference_engine.services.explainer.settings") as mock_settings:
        mock_settings.groq_api_key = ""
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
def mock_rf_model():
    model = MagicMock()
    model.predict_proba.return_value = [[0.1, 0.9]]
    model.feature_names_in_ = None
    return model


@pytest.mark.asyncio
async def test_explain_endpoint_returns_valid_schema(mock_rf_model):
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

    with patch("inference_engine.services.explainer.settings") as mock_settings:
        mock_settings.groq_api_key = ""
        result = await explain_risk(
            request=request,
            model=mock_rf_model,
            defiguard_model=None,
        )

    assert result.risk_level in ("low", "medium", "high")
    assert isinstance(result.reasons, list)
    assert len(result.reasons) >= 1
    assert isinstance(result.recommendation, str)
    assert result.exec_time_ms >= 0
