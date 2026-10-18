# -*- coding: utf-8 -*-
"""
inference_engine/api/main.py
==============================
FastAPI application entry point for the Web3 Transaction Firewall inference
engine — now powered by a **dual-model ensemble**.

Endpoints
---------
POST /predict           Score a transaction using the Random Forest + DeFiGuard
                        ensemble and return a detailed fraud verdict.
POST /predict/explain   Same as /predict plus an automated risk explanation.
POST /api/transactions/explain  Alias for /predict/explain.
GET  /health            Liveness probe — reports the status of both models.

Ensemble model loading
----------------------
At startup the engine loads **two** models:
1. **Random Forest** (required) — 200-tree sklearn classifier, 7 tabular
   features. The server enters a 503 degraded state if this fails.
2. **DeFiGuard** (optional) — a GNN + tabular fusion PyTorch neural network.
   If the file is absent or PyTorch is not installed, the engine falls back
   to RF-only mode automatically with a warning log.

The active ensemble strategy is configured via ``ENSEMBLE_MODE`` (default
``max`` — returns the higher of the two model scores for maximum recall).

Running
-------
    # From backend/
    uvicorn inference_engine.api.main:app --host 0.0.0.0 --port 8001 --reload
    # or
    python -m inference_engine.api.main
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import RandomForestClassifier

from inference_engine.api.dependencies import (
    get_app_state,
    get_defiguard_model,
    get_model,
    get_model_state,  # backward-compat alias
)
from inference_engine.api.schemas import (
    ExtractionRequest,
    HealthResponse,
    PredictionResponse,
    RiskExplanationRequest,
    RiskExplanationResponse,
    StatsResponse,
    TransactionFeatures,
)
from inference_engine.core.config import settings
from inference_engine.services.prediction import run_inference
from inference_engine.services.explainer import explain_transaction_risk

# Feature extractor — imported lazily to avoid hard-fail if backend/ not on sys.path
try:
    from feature_extractor.extractor import extract_features, ExtractionError
    _EXTRACTOR_AVAILABLE = True
except ImportError:
    _EXTRACTOR_AVAILABLE = False
    ExtractionError = ValueError  # type: ignore[misc,assignment]

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("inference_engine")


# --------------------------------------------------------------------------- #
# Lifespan: both models are loaded here — once — before any requests are served.
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load models on startup; clean up on shutdown."""
    app_state = get_app_state()

    # 1. Random Forest (required)
    logger.info("Startup: loading Random Forest model from %s", settings.model_path)
    try:
        app_state.load(settings.model_path)
        logger.info("Startup: RF ready. Fraud threshold: %.2f", settings.fraud_threshold)
    except FileNotFoundError as exc:
        logger.critical("STARTUP FAILED — RF model not found: %s", exc)
        # Allow server to start so /health can report the issue,
        # but /predict will return 503.

    # 2. DeFiGuard (optional — non-fatal failure)
    logger.info(
        "Startup: loading DeFiGuard model from %s (optional)", settings.defiguard_model_path
    )
    app_state.load_defiguard(settings.defiguard_model_path)

    if app_state.defiguard_loaded:
        logger.info(
            "Startup: ensemble mode = '%s' | RF + DeFiGuard both loaded.",
            settings.ensemble_mode,
        )
    else:
        logger.warning(
            "Startup: DeFiGuard model NOT loaded — running RF-only mode. "
            "Set DEFIGUARD_MODEL_PATH and ensure PyTorch is installed to enable the ensemble."
        )

    yield
    # Shutdown: nothing to close for joblib / PyTorch CPU models.
    logger.info("Shutdown: inference engine stopped.")


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="Web3 Transaction Firewall — Dual-Model Ensemble Inference Engine",
    description=(
        "Real-time fraud scoring API for Ethereum transactions. "
        "Accepts a DeFiTransLyzer feature vector and returns a fraud verdict "
        "from a **dual-model ensemble**: an AGA-optimised Random Forest classifier "
        "combined with the DeFiGuard GNN + tabular fusion neural network. "
        "The ensemble uses the ``max`` strategy by default (highest recall) — "
        "configurable via the ``ENSEMBLE_MODE`` environment variable."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — restrict to configured origins (defaults to "*" for dev; set CORS_ORIGINS in .env for prod).
_cors_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.post(
    "/predict/explain",
    response_model=RiskExplanationResponse,
    summary="Score transaction + generate plain-language risk explanation",
    tags=["Inference"],
)
@app.post(
    "/api/transactions/explain",
    response_model=RiskExplanationResponse,
    summary="Score transaction + risk explanation (alias)",
    tags=["Inference"],
)
async def explain_risk(
    request: RiskExplanationRequest,
    model: RandomForestClassifier = Depends(get_model),
    defiguard_model: Optional[Any] = Depends(get_defiguard_model),
) -> RiskExplanationResponse:
    """Score transaction risk via the ensemble and generate a plain-language explanation."""
    features_dict = request.features.model_dump()
    prediction = await run_inference(model, features_dict, defiguard_model=defiguard_model)
    get_app_state().record_prediction(prediction.is_fraud, prediction.exec_time_ms)

    explanation = await explain_transaction_risk(
        features=features_dict,
        is_fraud=prediction.is_fraud,
        fraud_probability=prediction.fraud_probability,
        raw_from=request.raw_from,
        raw_to=request.raw_to,
    )

    return RiskExplanationResponse(**explanation)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Score a transaction feature vector (ensemble)",
    tags=["Inference"],
)
async def predict(
    features: TransactionFeatures,
    model: RandomForestClassifier = Depends(get_model),
    defiguard_model: Optional[Any] = Depends(get_defiguard_model),
) -> PredictionResponse:
    """Receive a pre-extracted DeFiTransLyzer feature vector and return the
    ensemble fraud verdict:

    - **is_fraud**: `true` if the ensemble probability exceeds the configured threshold.
    - **fraud_probability**: Best-of-both-models confidence score (0–1).
    - **rf_score**: Raw Random Forest score.
    - **defiguard_score**: Raw DeFiGuard neural-network score (null if not loaded).
    - **ensemble_mode**: Active combination strategy (max | mean | weighted | rf_only).
    - **primary_model**: Which model had the higher / decisive score.
    - **exec_time_ms**: End-to-end ensemble inference latency in milliseconds.
    """
    result = await run_inference(model, features.model_dump(), defiguard_model=defiguard_model)
    get_app_state().record_prediction(result.is_fraud, result.exec_time_ms)
    return result


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness / readiness probe",
    tags=["Operations"],
)
async def health() -> HealthResponse:
    """Return the current health status of the inference engine including both models."""
    state = get_app_state()
    is_ready = state.is_loaded or (state.model is not None)

    # Determine active ensemble mode for display
    active_mode = settings.ensemble_mode
    if not state.defiguard_loaded:
        active_mode = "rf_only"

    return HealthResponse(
        status="ok" if is_ready else "degraded",
        model_loaded=is_ready,
        model_path=state.model_path,
        defiguard_loaded=state.defiguard_loaded,
        defiguard_model_path=state.defiguard_model_path,
        ensemble_mode=active_mode,
    )


@app.get(
    "/stats",
    response_model=StatsResponse,
    summary="Live inference telemetry since startup",
    tags=["Operations"],
)
async def stats() -> StatsResponse:
    """Return live counters: total predictions, fraud count, avg latency, uptime."""
    import datetime as _dt
    state = get_app_state()
    n = state.total_predictions
    fraud_rate = (state.total_fraud / n * 100) if n > 0 else 0.0
    avg_latency = (state.sum_latency_ms / n) if n > 0 else 0.0
    uptime = (_dt.datetime.now(_dt.timezone.utc) - state.started_at).total_seconds()
    active_mode = settings.ensemble_mode if state.defiguard_loaded else "rf_only"
    return StatsResponse(
        total_predictions=n,
        total_fraud=state.total_fraud,
        fraud_rate_pct=round(fraud_rate, 2),
        avg_latency_ms=round(avg_latency, 3),
        uptime_seconds=round(uptime, 1),
        ensemble_mode=active_mode,
    )


@app.post(
    "/extract",
    response_model=dict,
    summary="Extract DeFiTransLyzer features from a raw Ethereum transaction",
    tags=["Inference"],
)
async def extract(request: ExtractionRequest) -> dict:
    """Convert a raw JSON-RPC Ethereum transaction payload into the 20-feature
    DeFiTransLyzer vector consumed by /predict. Useful for testing the full
    pipeline from raw on-chain data.

    Raises 422 if the payload is malformed, 503 if the extractor module is
    unavailable (e.g. backend/ not on sys.path).
    """
    if not _EXTRACTOR_AVAILABLE:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail="Feature extractor module unavailable. Ensure backend/ is on PYTHONPATH.",
        )
    try:
        features = extract_features(request.tx_payload)
    except ExtractionError as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return features


# --------------------------------------------------------------------------- #
# Dev entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "inference_engine.api.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=False,
    )
