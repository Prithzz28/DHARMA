# -*- coding: utf-8 -*-
"""
inference_engine/services/prediction.py
=========================================
Core inference logic: converts a validated feature dict into a
``PredictionResponse`` using the **dual-model ensemble**.

Design decisions
-----------------
* Both models (Random Forest and DeFiGuard) are scored **concurrently** via
  ``asyncio`` + ``ThreadPoolExecutor`` to keep end-to-end latency low.
* The final fraud probability is determined by the active ``ENSEMBLE_MODE``
  setting (default: ``max`` — highest recall).
* The ``PredictionResponse`` now includes per-model breakdown fields
  (``rf_score``, ``defiguard_score``, ``ensemble_mode``, ``primary_model``)
  so callers can inspect the individual model contributions.
* If the DeFiGuard model is unavailable, the engine silently falls back to
  RF-only mode — ``defiguard_score`` will be ``null`` in the response.
* Timing is measured end-to-end (dict → response) to include numpy
  conversion and async scheduling overhead.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from inference_engine.api.schemas import PredictionResponse
from inference_engine.core.config import settings
from inference_engine.services.ensemble import EnsembleMode, ensemble_predict

logger = logging.getLogger(__name__)

# The trained Random Forest model uses exactly these 7 features (in this order).
# At runtime, ensemble.py prefers model.feature_names_in_ if available;
# this constant serves as the documented fallback and is imported by ensemble.py.
FEATURE_ORDER: Sequence[str] = (
    "length_transaction_hash",
    "length_to",
    "block_number",
    "gas_efficiency",
    "chain_id",
    "effective_gas_price",
    "is_same_address",
)
# NOTE: All actual model inference is delegated to ensemble.py which maintains
# its own ThreadPoolExecutor. There is intentionally no executor here.


def _build_feature_array(features: Dict[str, float]) -> np.ndarray:
    """Convert a dictionary of features into a 2D float32 numpy array matching FEATURE_ORDER."""
    values = [float(features.get(k, 0.0) or 0.0) for k in FEATURE_ORDER]
    return np.array([values], dtype=np.float32)


async def run_inference(
    model: RandomForestClassifier,
    features: Dict[str, float],
    defiguard_model: Optional[Any] = None,
) -> PredictionResponse:
    """Asynchronously score a transaction feature dictionary using the ensemble.

    Parameters
    ----------
    model:
        Loaded ``RandomForestClassifier`` instance equipped with ``predict_proba``.
    features:
        Mapping of feature names to numeric values.
    defiguard_model:
        Optional loaded ``DeFiGuardPredictor`` instance. When provided, both
        models are scored concurrently and their results are combined via the
        active ``ENSEMBLE_MODE`` setting (default: ``max``).

    Returns
    -------
    PredictionResponse
        Structured prediction response containing:
        - ``is_fraud``          — ensemble verdict.
        - ``fraud_probability`` — ensemble score (best-of-both).
        - ``rf_score``          — raw Random Forest score.
        - ``defiguard_score``   — raw DeFiGuard score (None if unavailable).
        - ``ensemble_mode``     — active combination strategy.
        - ``primary_model``     — which model produced the highest / decisive score.
        - ``exec_time_ms``      — end-to-end latency in milliseconds.

    Raises
    ------
    TypeError
        If model or features are of invalid types.
    ValueError
        If model is uninitialized.
    RuntimeError
        If prediction execution fails.
    """
    if model is None:
        raise ValueError("Model instance cannot be None")
    if not hasattr(model, "predict_proba") or not callable(model.predict_proba):
        raise TypeError("Model must be a valid classifier implementing 'predict_proba'")
    if features is None or not isinstance(features, dict):
        raise TypeError(
            f"Expected features dictionary, got "
            f"{type(features).__name__ if features is not None else 'None'}"
        )

    start_time = time.perf_counter()

    # Parse ensemble mode from settings
    try:
        mode = EnsembleMode(settings.ensemble_mode.lower())
    except ValueError:
        logger.warning(
            "Unknown ENSEMBLE_MODE '%s' — defaulting to 'max'.", settings.ensemble_mode
        )
        mode = EnsembleMode.MAX

    # --- Run ensemble (concurrent RF + DeFiGuard) ---
    try:
        final_prob, rf_score, dg_score, active_mode = await ensemble_predict(
            rf_model=model,
            dg_model=defiguard_model,
            features=features,
            mode=mode,
            rf_weight=settings.rf_weight,
            dg_weight=settings.dg_weight,
        )
    except Exception as exc:
        logger.error("Ensemble inference failed: %s", exc)
        raise

    execution_latency_ms = (time.perf_counter() - start_time) * 1_000
    is_fraud = final_prob > settings.fraud_threshold

    # Determine which model was the primary contributor
    if dg_score is None:
        primary_model = "random_forest"
    elif active_mode == "dg_only":
        primary_model = "defiguard"
    elif active_mode == "rf_only":
        primary_model = "random_forest"
    else:
        primary_model = "defiguard" if dg_score >= rf_score else "random_forest"

    logger.debug(
        "Ensemble [%s] final=%.4f rf=%.4f dg=%s is_fraud=%s latency=%.2f ms",
        active_mode,
        final_prob,
        rf_score,
        f"{dg_score:.4f}" if dg_score is not None else "N/A",
        is_fraud,
        execution_latency_ms,
    )

    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_probability=round(final_prob, 6),
        exec_time_ms=round(execution_latency_ms, 3),
        rf_score=round(rf_score, 6),
        defiguard_score=round(dg_score, 6) if dg_score is not None else None,
        ensemble_mode=active_mode,
        primary_model=primary_model,
    )
