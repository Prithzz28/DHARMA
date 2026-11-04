# -*- coding: utf-8 -*-
"""
inference_engine/api/main.py
==============================
FastAPI application entry point for the Web3 Transaction Firewall inference
engine, powered by the **DeFiGuard** model (GATv2 graph encoder + tabular
encoder with cross-attention fusion).

Endpoints
---------
POST /scan              Score a pending transaction from user-level parameters
                        (server-side feature extraction → DeFiGuard → optional
                        explanation). Used by the dashboard.
POST /predict           Score a raw JSON-RPC transaction, optionally with its
                        receipt (pending vs receipt mode).
POST /predict/explain   /predict plus a risk explanation (response includes the
                        prediction). Alias: POST /api/transactions/explain.
POST /extract           Raw JSON-RPC payload → DeFiGuard feature vector.
POST /rpc               Wallet JSON-RPC endpoint: transaction submissions are scored
                        and blocked / flagged / forwarded; other calls are proxied
                        to UPSTREAM_RPC_URL. GET /rpc/info, GET /rpc/recent.
GET  /health            Liveness / readiness — model status and thresholds.
GET  /stats             Live counters since startup.

Scoring endpoints (/scan, /predict*, /extract) are rate limited per client IP
(``RATE_LIMIT_PER_MINUTE``). If the model fails to load the server still starts
so /health can report ``degraded``; scoring endpoints then return 503.

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
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from feature_extractor.extractor import ExtractionError, extract_features, split_payload
from inference_engine.api.dependencies import get_app_state, get_defiguard_model
from inference_engine.api.rpc import router as rpc_router
from inference_engine.api.schemas import (
    ExtractionRequest,
    ExtractionResponse,
    HealthResponse,
    PredictRequest,
    PredictionResponse,
    RiskExplanationRequest,
    RiskExplanationResponse,
    ScanRequest,
    ScanResponse,
    StatsResponse,
)
from inference_engine.core.config import obsolete_settings_in_use, settings
from inference_engine.core.rate_limit import rate_limited
from inference_engine.services.explainer import explain_transaction_risk
from inference_engine.services.prediction import run_inference, threshold_for
from inference_engine.services.rpc_firewall import close_upstream_client

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("inference_engine")


# --------------------------------------------------------------------------- #
# Lifespan: the model is loaded once, before any requests are served.
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load DeFiGuard on startup."""
    obsolete = obsolete_settings_in_use()
    if obsolete:
        logger.warning(
            "Ignoring obsolete settings from the removed Random Forest ensemble: %s. "
            "Remove them from your environment / .env.",
            ", ".join(obsolete),
        )

    logger.info("Startup: loading DeFiGuard model from %s", settings.defiguard_model_path)
    try:
        get_app_state().load_defiguard(settings.defiguard_model_path)
        logger.info("Startup: ready. Threshold override: %s", settings.fraud_threshold)
    except Exception as exc:  # noqa: BLE001 — start degraded so /health can report it
        logger.critical(
            "STARTUP DEGRADED — DeFiGuard model could not be loaded (%s: %s). "
            "Scoring endpoints will return 503.",
            type(exc).__name__,
            exc,
        )
    yield
    await close_upstream_client()
    logger.info("Shutdown: inference engine stopped.")


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="Web3 Transaction Firewall — DeFiGuard Inference Engine",
    description=(
        "Real-time fraud scoring API for Ethereum transactions. Pending "
        "transactions are scored from pre-execution features; transactions "
        "with a receipt are scored by the full DeFiGuard model (GATv2 receipt "
        "ego-graph + tabular encoder with cross-attention fusion). Block "
        "thresholds ship with the model; ``FRAUD_THRESHOLD`` overrides them."
    ),
    version="3.0.0",
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
    expose_headers=["Retry-After"],
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """422 with field locations and messages only. The rejected input is not
    echoed back — it may be NaN/Infinity, which is not valid JSON."""
    errors = [
        {"loc": list(err.get("loc", ())), "msg": err.get("msg", ""), "type": err.get("type", "")}
        for err in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


app.include_router(rpc_router)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
def _scan_to_tx(req: ScanRequest) -> Dict[str, Any]:
    """Convert user-level scan parameters into a JSON-RPC pending transaction."""
    data = req.data.strip()
    return {
        "from": req.from_address.strip() or None,
        "to": req.to_address.strip() or None,
        "value": hex(int(round(req.value_eth * 1e18))),
        "gas": hex(req.gas_limit),
        "gasPrice": hex(int(round(req.gas_price_gwei * 1e9))),
        "nonce": hex(req.nonce),
        "input": data if data.startswith("0x") else "0x" + data,
        "chainId": hex(req.chain_id),
    }


async def _score(model: Any, tx: Dict[str, Any], receipt: Any) -> PredictionResponse:
    """run_inference with malformed input reported as 422."""
    try:
        prediction = await run_inference(model, tx, receipt)
    except ExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    get_app_state().record_prediction(prediction.is_fraud, prediction.exec_time_ms)
    return prediction


@app.post(
    "/scan",
    response_model=ScanResponse,
    summary="Score a transaction from user-level parameters (extract → DeFiGuard → explain)",
    tags=["Inference"],
    dependencies=[Depends(rate_limited)],
)
async def scan(
    request: ScanRequest,
    model: Any = Depends(get_defiguard_model),
) -> ScanResponse:
    """Build a pending transaction from the given parameters, extract the
    pre-execution features server-side, score them with DeFiGuard (pending
    mode) and, when ``explain`` is true, generate a risk explanation —
    rule-based unless ``use_llm`` is true and a Groq key is configured.
    """
    tx = _scan_to_tx(request)
    features = extract_features(tx)
    prediction = await _score(model, tx, None)

    explanation = None
    if request.explain:
        explanation = RiskExplanationResponse(
            **await explain_transaction_risk(
                features=features,
                is_fraud=prediction.is_fraud,
                fraud_probability=prediction.fraud_probability,
                threshold=prediction.threshold,
                raw_from=tx["from"],
                raw_to=tx["to"],
                use_llm=request.use_llm,
            )
        )
    return ScanResponse(features=features, prediction=prediction, explanation=explanation)


@app.post(
    "/predict/explain",
    response_model=RiskExplanationResponse,
    summary="Score transaction + generate plain-language risk explanation",
    tags=["Inference"],
    dependencies=[Depends(rate_limited)],
)
@app.post(
    "/api/transactions/explain",
    response_model=RiskExplanationResponse,
    summary="Score transaction + risk explanation (alias)",
    tags=["Inference"],
    dependencies=[Depends(rate_limited)],
)
async def explain_risk(
    request: RiskExplanationRequest,
    model: Any = Depends(get_defiguard_model),
) -> RiskExplanationResponse:
    """Score transaction risk with DeFiGuard and generate a plain-language
    explanation. The response includes the ``prediction`` it is based on."""
    prediction = await _score(model, request.tx, request.receipt)
    features = extract_features({"tx": request.tx, "receipt": request.receipt})
    explanation = await explain_transaction_risk(
        features=features,
        is_fraud=prediction.is_fraud,
        fraud_probability=prediction.fraud_probability,
        threshold=prediction.threshold,
        raw_from=request.tx.get("from"),
        raw_to=request.tx.get("to"),
        use_llm=request.use_llm,
    )
    return RiskExplanationResponse(**explanation, prediction=prediction)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Score a raw transaction (optionally with its receipt)",
    tags=["Inference"],
    dependencies=[Depends(rate_limited)],
)
async def predict(
    request: PredictRequest,
    model: Any = Depends(get_defiguard_model),
) -> PredictionResponse:
    """Score a raw JSON-RPC transaction and return the verdict:

    - **mode**: `pending` (no receipt: pre-execution features) or `receipt`
      (full DeFiGuard over the receipt ego-graph).
    - **is_fraud**: `true` if `fraud_probability` reaches `threshold`.
    - **fraud_probability**: DeFiGuard fraud probability (0–1).
    - **threshold**: The block threshold for this mode.
    - **exec_time_ms**: Inference latency in milliseconds.
    """
    return await _score(model, request.tx, request.receipt)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness / readiness probe",
    tags=["Operations"],
)
async def health() -> HealthResponse:
    """Return the current health status of the inference engine."""
    state = get_app_state()
    return HealthResponse(
        status="ok" if state.defiguard_loaded else "degraded",
        model_loaded=state.defiguard_loaded,
        model_path=state.defiguard_model_path,
        thresholds=(
            {m: threshold_for(state.defiguard_model, m) for m in ("pending", "receipt")}
            if state.defiguard_loaded else {}
        ),
        llm_configured=bool(settings.groq_api_key),
        llm_model=settings.groq_model,
    )


@app.get(
    "/stats",
    response_model=StatsResponse,
    summary="Live inference telemetry since startup",
    tags=["Operations"],
)
async def stats() -> StatsResponse:
    """Return live counters: total predictions, fraud count, avg latency, uptime."""
    state = get_app_state()
    n = state.total_predictions
    fraud_rate = (state.total_fraud / n * 100) if n > 0 else 0.0
    avg_latency = (state.sum_latency_ms / n) if n > 0 else 0.0
    uptime = (datetime.now(timezone.utc) - state.started_at).total_seconds()
    return StatsResponse(
        total_predictions=n,
        total_fraud=state.total_fraud,
        fraud_rate_pct=round(fraud_rate, 2),
        avg_latency_ms=round(avg_latency, 3),
        uptime_seconds=round(uptime, 1),
    )


@app.post(
    "/extract",
    response_model=ExtractionResponse,
    summary="Extract DeFiGuard features from a raw Ethereum transaction",
    tags=["Inference"],
    dependencies=[Depends(rate_limited)],
)
async def extract(request: ExtractionRequest) -> ExtractionResponse:
    """Convert a raw JSON-RPC payload into DeFiGuard's feature vector (receipt
    features are 0 in pending mode). Raises 422 if malformed."""
    try:
        _, receipt = split_payload(request.tx_payload)
        features = extract_features(request.tx_payload)
    except ExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ExtractionResponse(mode="receipt" if receipt else "pending", features=features)


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
