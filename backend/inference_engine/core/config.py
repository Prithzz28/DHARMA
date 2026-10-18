# -*- coding: utf-8 -*-
"""
inference_engine/core/config.py
================================
Centralised settings for the inference engine, loaded from environment
variables (with sensible defaults) via Pydantic Settings.

Environment variables
---------------------
MODEL_PATH          Absolute or relative path to the Random Forest .joblib file.
DEFIGUARD_MODEL_PATH  Absolute or relative path to the DeFiGuard .joblib file.
ENSEMBLE_MODE       Combination strategy: max | mean | weighted | rf_only | dg_only.
                    Defaults to ``max`` (highest recall — best for fraud prevention).
RF_WEIGHT           RF weight used in ``weighted`` ensemble mode (default 0.45).
DG_WEIGHT           DeFiGuard weight used in ``weighted`` ensemble mode (default 0.55).
FRAUD_THRESHOLD     Float in (0, 1). Predictions above this are flagged as fraud.
HOST                Host to bind the Uvicorn server to.
PORT                Port to bind the Uvicorn server to.
LOG_LEVEL           Python logging level string (INFO, DEBUG, WARNING …).

Usage
-----
    from inference_engine.core.config import settings
    print(settings.model_path)
    print(settings.defiguard_model_path)
    print(settings.ensemble_mode)
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Models
    # ------------------------------------------------------------------ #
    #: Path to the serialised Random Forest model (.joblib).
    model_path: Path = Path("models/random_forest_fraud_model.joblib")

    #: Path to the serialised DeFiGuard neural-network model (.joblib).
    #: When set (and the file exists), the engine runs BOTH models in
    #: parallel and combines their predictions via the ensemble strategy.
    defiguard_model_path: Path = Path("models/defi_guard_model.joblib")

    # ------------------------------------------------------------------ #
    # Ensemble configuration
    # ------------------------------------------------------------------ #
    #: How to combine the two model scores.
    #: ``max``      — use the higher score (highest recall, recommended).
    #: ``mean``     — arithmetic mean of both scores.
    #: ``weighted`` — weighted average (see rf_weight / dg_weight below).
    #: ``rf_only``  — disable DeFiGuard, use RF only.
    #: ``dg_only``  — use DeFiGuard only (not recommended for production).
    ensemble_mode: str = "max"

    #: Weights for ``weighted`` ensemble mode (must sum to > 0).
    #: Empirically calibrated: RF (0.55) provides high precision on tabular features,
    #: while DeFiGuard (0.45) provides deep structural fraud signals.
    rf_weight: float = 0.55
    dg_weight: float = 0.45

    #: Fraud probability above which a transaction is *blocked*.
    fraud_threshold: float = 0.65

    # ------------------------------------------------------------------ #
    # Server
    # ------------------------------------------------------------------ #
    host: str = "0.0.0.0"
    port: int = 8001
    log_level: str = "INFO"

    #: Comma-separated list of allowed CORS origins.
    #: Set to the frontend URL (e.g. "http://localhost:5173") for production.
    #: Defaults to "*" for development convenience only.
    cors_origins: str = "*"

    # ------------------------------------------------------------------ #
    # Risk Explanation (LLM)
    # ------------------------------------------------------------------ #
    groq_api_key: str = ""


#: Module-level singleton — import this everywhere instead of re-creating.
settings = Settings()
