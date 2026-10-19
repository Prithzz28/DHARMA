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

from typing import Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

#: Upper bound for address / hash strings accepted from clients.
MAX_ID_LENGTH = 100


class TransactionFeatures(BaseModel):
    """The 20-dimensional feature vector expected by the model.

    All fields are optional at the HTTP level (default 0.0) so that callers
    that cannot derive every metric still receive a prediction rather than a
    validation error — partial feature vectors are preferable to dropped
    requests in a live firewall. NaN and ±Infinity are rejected.
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
        default=0.0, ge=0.0,
        description="cumulative_gas_used / gas_used (DeFiTransLyzer definition; 0 when unknown).",
    )
    effective_gas_price: float = Field(
        default=0.0, ge=0.0, description="Effective gas price paid (wei)."
    )
    total_gas_cost: float = Field(
        default=0.0, ge=0.0, description="gas_used × effective_gas_price (wei)."
    )
    gas_price_ratio: float = Field(
        default=1.0, ge=0.0,
        description="Paid ÷ offered gas price; 1.0 for pending transactions (as in training).",
    )
    gas_per_log_event: float = Field(
        default=0.0, ge=0.0, description="Constant 0 in the training data."
    )

    # -- Value -------------------------------------------------------------- #
    value: float = Field(
        default=0.0, ge=0.0, description="ETH value transferred (wei)."
    )
    normalized_token_transfer: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Constant 0 in the training data."
    )

    # -- Log / event activity ----------------------------------------------- #
    log_count: float = Field(
        default=0.0, ge=0.0, description="Constant 0 in the training data (see length_log)."
    )
    log_removed: float = Field(
        default=0.0, ge=0.0, le=1.0, description="1.0 if any logs were removed (reorg flag)."
    )
    length_log: float = Field(
        default=0.0, ge=0.0, description="Number of log entries emitted (DeFiTransLyzer definition)."
    )
    event_activity_flag: float = Field(
        default=0.0, ge=0.0, le=1.0, description="1.0 if the transaction emitted any logs."
    )

    model_config = ConfigDict(allow_inf_nan=False, json_schema_extra={
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
            "gas_price_ratio": 1.0,
            "length_log": 0,
            "value": 1_000_000_000_000_000_000,
        }
    })


class PredictionResponse(BaseModel):
    """DeFiGuard verdict returned by /predict, /scan and /predict/explain."""

    is_fraud: bool = Field(description="True if fraud_probability exceeds the threshold.")
    fraud_probability: float = Field(ge=0.0, le=1.0, description="DeFiGuard fraud probability (0–1).")
    threshold: float = Field(description="Block threshold applied (FRAUD_THRESHOLD).")
    model: Literal["defiguard"] = Field(default="defiguard", description="Model that produced the score.")
    exec_time_ms: float = Field(ge=0.0, description="Inference latency in milliseconds.")


class HealthResponse(BaseModel):
    """Response body for the /health liveness probe."""

    status: Literal["ok", "degraded"]
    model_loaded: bool = Field(description="True if the DeFiGuard model is loaded.")
    model_path: str = Field(default="", description="Resolved path of the loaded model file.")
    fraud_threshold: float = Field(description="Probability above which a transaction is blocked.")
    llm_configured: bool = Field(default=False, description="True if a Groq API key is set.")
    llm_model: str = Field(default="", description="Groq model used for AI explanations.")


class RiskExplanationRequest(BaseModel):
    """Payload sent to POST /predict/explain or POST /api/transactions/explain."""

    features: TransactionFeatures = Field(default_factory=TransactionFeatures)
    model_config = ConfigDict(allow_inf_nan=False)

    raw_from: Optional[str] = Field(
        default=None, max_length=MAX_ID_LENGTH, description="Optional raw sender address (masked before use)."
    )
    raw_to: Optional[str] = Field(
        default=None, max_length=MAX_ID_LENGTH, description="Optional raw recipient address (masked before use)."
    )
    amount_eth: Optional[float] = Field(default=None, ge=0.0, description="Optional ETH amount transferred.")


class RiskExplanationResponse(BaseModel):
    """Structured AI risk explanation result."""

    risk_level: Literal["low", "medium", "high"]
    reasons: list[str] = Field(min_length=1, description="Bullet points explaining the verdict.")
    recommendation: str = Field(description="Actionable security recommendation for the user.")
    exec_time_ms: float = Field(ge=0.0, description="Latency of the explainer engine in ms.")
    is_fallback: bool = Field(default=True, description="True if generated by rule fallback engine.")
    fallback_reason: Optional[str] = Field(
        default=None,
        description="Why an AI explanation was requested but not produced (None otherwise).",
    )
    prediction: Optional[PredictionResponse] = Field(
        default=None, description="The prediction the explanation is based on."
    )


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
    avg_latency_ms: float = Field(ge=0.0, description="Average inference latency (ms).")
    uptime_seconds: float = Field(ge=0.0, description="Seconds since the engine started.")


class ExtractionRequest(BaseModel):
    """Raw Ethereum transaction payload for POST /extract."""

    tx_payload: dict = Field(
        description=(
            "Raw JSON-RPC Ethereum transaction or receipt object. "
            "Accepts eth_sendTransaction, eth_getTransactionReceipt, or mixed formats."
        )
    )


class ScanRequest(BaseModel):
    """User-level transaction parameters for POST /scan.

    The server converts these into a JSON-RPC style transaction, extracts the
    DeFiTransLyzer features (``feature_extractor``), scores them and optionally
    explains the verdict — so feature engineering lives in one place.
    """

    model_config = ConfigDict(allow_inf_nan=False, populate_by_name=True)

    from_address: str = Field(default="", max_length=MAX_ID_LENGTH, alias="from")
    to_address: str = Field(default="", max_length=MAX_ID_LENGTH, alias="to")
    value_eth: float = Field(default=0.0, ge=0.0, le=1e9, description="Amount transferred (ETH).")
    gas_price_gwei: float = Field(gt=0.0, le=1e6, description="Gas price / max fee (Gwei).")
    gas_limit: int = Field(default=21_000, gt=0, le=100_000_000, description="Gas limit (units).")
    block_number: int = Field(default=0, ge=0, le=10**10)
    tx_hash: str = Field(default="", max_length=MAX_ID_LENGTH)
    log_count: int = Field(default=0, ge=0, le=10_000)
    chain_id: int = Field(default=1, ge=0, le=2**32)
    nonce: int = Field(default=0, ge=0, le=10**12)
    explain: bool = Field(default=True, description="Also generate a risk explanation.")
    use_llm: bool = Field(
        default=False, description="Let the explanation use the Groq LLM (if configured); otherwise rule-based."
    )


class ScanResponse(BaseModel):
    """Extracted features, prediction and optional explanation."""

    features: Dict[str, float]
    prediction: PredictionResponse
    explanation: Optional[RiskExplanationResponse] = None
ScanRequest.model_rebuild()
ScanResponse.model_rebuild()
