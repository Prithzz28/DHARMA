# -*- coding: utf-8 -*-
"""
inference_engine/services/prediction.py
=========================================
Core inference: scores a feature dict with the DeFiGuard model and returns a
``PredictionResponse``.

* The PyTorch forward pass is blocking, so it runs in a small thread pool to
  keep the event loop responsive.
* ``is_fraud`` is ``fraud_probability > FRAUD_THRESHOLD``.
* A non-finite model output raises ``ValueError`` (surfaced as HTTP 500) — it
  is never masked as a neutral score.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from inference_engine.api.schemas import PredictionResponse
from inference_engine.core.config import settings

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="defiguard_inference")


def _sync_predict(model: Any, features: Dict[str, float], raw_tx: Optional[Dict[str, Any]]) -> float:
    """Blocking DeFiGuard call (graph + tabular forward pass) — run inside the executor."""
    return float(model.predict_fraud_proba(features, raw_tx))


async def run_inference(
    model: Any,
    features: Dict[str, float],
    raw_tx: Optional[Dict[str, Any]] = None,
) -> PredictionResponse:
    """Score a transaction feature dictionary with DeFiGuard.

    Parameters
    ----------
    model:
        Loaded ``DeFiGuardPredictor``.
    features:
        Mapping of feature names to numeric values.
    raw_tx:
        Optional raw transaction fields (nonce/logs) for DeFiGuard's ego-graph.
        ``status`` is left unset (0), as for every transaction in the training data.
    """
    if model is None:
        raise ValueError("Model instance cannot be None")
    if not isinstance(features, dict):
        raise TypeError(f"Expected features dictionary, got {type(features).__name__}")

    start_time = time.perf_counter()
    loop = asyncio.get_running_loop()
    probability = await loop.run_in_executor(_executor, _sync_predict, model, features, raw_tx)
    if not math.isfinite(probability):
        raise ValueError(f"DeFiGuard returned a non-finite probability: {probability!r}")
    probability = min(1.0, max(0.0, probability))

    latency_ms = (time.perf_counter() - start_time) * 1_000
    threshold = settings.fraud_threshold
    is_fraud = probability > threshold
    logger.debug("DeFiGuard p=%.4f threshold=%.2f is_fraud=%s latency=%.2f ms",
                 probability, threshold, is_fraud, latency_ms)

    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_probability=round(probability, 6),
        threshold=threshold,
        exec_time_ms=round(latency_ms, 3),
    )
