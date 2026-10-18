# -*- coding: utf-8 -*-
"""
inference_engine/api/schemas.py
================================
Pydantic models for strict request/response validation on the /predict
endpoint.

Feature names mirror ``data_loader.TRANSACTION_FEATURES`` exactly so that
the payload maps 1-to-1 to the columns the Random Forest was trained on.
All 20 features are floats; missing fields fall back to 0.0 which is the
safe median-imputed default used during training preprocessing.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class TransactionFeatures(BaseModel):
    """The 20-dimensional feature vector expected by the model.

    All fields are optional at the HTTP level (default 0.0) so that callers
    that cannot derive every metric still receive a prediction rather than a
    validation error — partial feature vectors are preferable to dropped
    requests in a live firewall.
    """

    # -- Identity / structural --------------------------------------------- #
    length_transaction_hash: float = Field(
        default=0.0, ge=0.0, description="Length of the transaction hash string."
    )
    length_to: float = Field(
        default=0.0, ge=0.0, description="Length of the recipient address string."
    )
    length_from: float = Field(
        default=0.0, ge=0.0, description="Length of the sender address string."
    )
    is_same_address: float = Field(
        default=0.0, ge=0.0, le=1.0, description="1.0 if sender == recipient."
    )
    index: float = Field(
        default=0.0, ge=0.0, description="Transaction index within its block."
    )

    # -- Block context ------------------------------------------------------ #
    block_number: float = Field(
        default=0.0, ge=0.0, description="Block height at which the transaction was included."
    )
    chain_id: float = Field(
        default=1.0, ge=0.0, description="Ethereum network chain ID (1 = mainnet)."
    )
    cumulative_gas_used: float = Field(
        default=0.0, ge=0.0, description="Cumulative gas consumed up to this tx in the block."
    )

    # -- Gas metrics -------------------------------------------------------- #
    gas_used: float = Field(
        default=0.0, ge=0.0, description="Gas actually consumed by this transaction."
    )
    gas_efficiency: float = Field(
        default=0.0, description="gas_used / gas_limit ratio."
    )
    effective_gas_price: float = Field(
        default=0.0, ge=0.0, description="Effective gas price paid (wei)."
    )
    total_gas_cost: float = Field(
        default=0.0, ge=0.0, description="gas_used × effective_gas_price (wei)."
    )
    gas_price_ratio: float = Field(
        default=0.0, ge=0.0, description="effective_gas_price / base_fee_per_gas."
    )
    gas_per_log_event: float = Field(
        default=0.0, ge=0.0, description="gas_used / (log_count + 1)."
    )

    # -- Value -------------------------------------------------------------- #
    value: float = Field(
        default=0.0, ge=0.0, description="ETH value transferred (wei)."
    )
    normalized_token_transfer: float = Field(
        default=0.0, description="Normalised transfer value."
    )

    # -- Log / event activity ----------------------------------------------- #
    log_count: float = Field(
        default=0.0, ge=0.0, description="Raw number of log entries emitted."
    )
    log_removed: float = Field(
        default=0.0, ge=0.0, le=1.0, description="1.0 if any logs were removed (reorg flag)."
    )
    length_log: float = Field(
        default=0.0, ge=0.0, description="log(log_count + 1) — estimated event generation."
    )
    event_activity_flag: float = Field(
        default=0.0, ge=0.0, le=1.0, description="1.0 if the transaction emitted any logs."
    )

    model_config = {"json_schema_extra": {
        "example": {
            "length_transaction_hash": 66,
            "length_to": 42,
            "length_from": 42,
            "block_number": 19_500_000,
            "gas_used": 21_000,
            "chain_id": 1,
            "total_gas_cost": 4_200_000_000_000,
            "effective_gas_price": 200_000_000_000,
            "cumulative_gas_used": 500_000,
            "gas_price_ratio": 2.5,
            "length_log": 0.693,
            "value": 1_000_000_000_000_000_000,
        }
    }}


class PredictionResponse(BaseModel):
    """Inference result returned by the /predict endpoint.

    The ``fraud_probability`` is the **ensemble** score — the best prediction
    from both the Random Forest and DeFiGuard models according to the active
    ensemble strategy (default: ``max`` of both scores).
    """

    is_fraud: bool = Field(
        description="True if the ensemble fraud probability exceeds the configured threshold."
    )
    fraud_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Ensemble fraud confidence score (0–1). Best-of-both-models result.",
    )
    exec_time_ms: float = Field(
        ge=0.0, description="End-to-end ensemble inference latency in milliseconds."
    )

    # ----- Per-model breakdown ----- #
    rf_score: float = Field(
        ge=0.0,
        le=1.0,
        default=0.0,
        description="Raw Random Forest fraud probability score.",
    )
    defiguard_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Raw DeFiGuard neural-network fraud probability score (None if model not loaded).",
    )
    ensemble_mode: str = Field(
        default="rf_only",
        description="Active ensemble combination strategy: max | mean | weighted | rf_only | dg_only.",
    )
    primary_model: str = Field(
        default="random_forest",
        description="Which model contributed the decisive (highest) score.",
    )


class HealthResponse(BaseModel):
    """Response body for the /health liveness probe."""

    status: str
    model_loaded: bool
    model_path: str
    defiguard_loaded: bool = Field(
        default=False,
        description="True if the DeFiGuard model is loaded and contributing to ensemble scoring.",
    )
    defiguard_model_path: str = Field(
        default="",
        description="Resolved path to the loaded DeFiGuard model file.",
    )
    ensemble_mode: str = Field(
        default="rf_only",
        description="Active ensemble strategy.",
    )


class RiskExplanationRequest(BaseModel):
    """Payload sent to POST /predict/explain or POST /api/transactions/explain."""

    features: TransactionFeatures = Field(default_factory=TransactionFeatures)
    raw_from: Optional[str] = Field(default=None, description="Optional raw sender address for PII masking.")
    raw_to: Optional[str] = Field(default=None, description="Optional raw recipient address for PII masking.")
    amount_eth: Optional[float] = Field(default=None, description="Optional ETH amount transferred.")


class RiskExplanationResponse(BaseModel):
    """Structured AI risk explanation result."""

    risk_level: str = Field(description="'low' | 'medium' | 'high'")
    reasons: list[str] = Field(description="Bullet points explaining why the transaction is flagged.")
    recommendation: str = Field(description="Actionable security recommendation for the user.")
    exec_time_ms: float = Field(ge=0.0, description="Latency of the explainer engine in ms.")
    is_fallback: bool = Field(default=True, description="True if generated by rule fallback engine.")


# Ensure Pydantic v2 type adapter resolves nested schemas under 'from __future__ import annotations'
RiskExplanationRequest.model_rebuild()
RiskExplanationResponse.model_rebuild()


class StatsResponse(BaseModel):
    """Live inference engine telemetry returned by GET /stats."""

    total_predictions: int = Field(description="Total predictions served since startup.")
    total_fraud: int = Field(description="Number of transactions flagged as fraud.")
    fraud_rate_pct: float = Field(
        ge=0.0, le=100.0, description="Fraud rate as a percentage of all predictions."
    )
    avg_latency_ms: float = Field(ge=0.0, description="Average ensemble inference latency (ms).")
    uptime_seconds: float = Field(ge=0.0, description="Seconds since the engine started.")
    ensemble_mode: str = Field(description="Currently active ensemble mode.")


class ExtractionRequest(BaseModel):
    """Raw Ethereum transaction payload for POST /extract."""

    tx_payload: dict = Field(
        description=(
            "Raw JSON-RPC Ethereum transaction or receipt object. "
            "Accepts eth_sendTransaction, eth_getTransactionReceipt, or mixed formats."
        )
    )
