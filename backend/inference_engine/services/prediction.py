# -*- coding: utf-8 -*-
"""
inference_engine/services/prediction.py
=========================================
Core inference: scores a raw transaction (plus optional receipt) with the
DeFiGuard predictor and returns a ``PredictionResponse``.

* No receipt → *pending* mode (pre-execution features); receipt → *receipt*
  mode (full DeFiGuard with the receipt ego-graph).
* The PyTorch forward pass is blocking, so it runs in a small thread pool to
  keep the event loop responsive.
* ``is_fraud`` is ``fraud_probability >= threshold``, where the threshold is
  the one stored in the model for that mode (selected on validation data in
  training) unless ``FRAUD_THRESHOLD`` overrides it.
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

from feature_extractor.extractor import ExtractionError
from inference_engine.api.schemas import PredictionResponse
from inference_engine.core.config import settings

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="defiguard_inference")


def threshold_for(model: Any, mode: str) -> float:
    """Block threshold for *mode*: the FRAUD_THRESHOLD override, else the model's."""
    if settings.fraud_threshold is not None:
        return float(settings.fraud_threshold)
    return float(model.thresholds[mode])


def _sync_predict(model: Any, tx: Dict[str, Any], receipt: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Blocking DeFiGuard call (features + graph + forward pass) — run inside the executor."""
    try:
        return model.score(tx, receipt)
    except (TypeError, AttributeError) as exc:
        raise ExtractionError(f"Malformed transaction or receipt: {exc}") from exc


async def run_inference(
    model: Any,
    tx: Dict[str, Any],
    receipt: Optional[Dict[str, Any]] = None,
) -> PredictionResponse:
    """Score one transaction with DeFiGuard.

    Parameters
    ----------
    model:
        Loaded ``DeFiGuardPredictor``.
    tx:
        JSON-RPC transaction object.
    receipt:
        Optional JSON-RPC receipt; selects receipt mode when given.
    """
    if model is None:
        raise ValueError("Model instance cannot be None")
    if not isinstance(tx, dict):
        raise TypeError(f"Expected transaction dict, got {type(tx).__name__}")

    start_time = time.perf_counter()
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(_executor, _sync_predict, model, tx, receipt or None)
    probability = result["fraud_probability"]
    if not math.isfinite(probability):
        raise ValueError(f"DeFiGuard returned a non-finite probability: {probability!r}")
    probability = min(1.0, max(0.0, probability))

    latency_ms = (time.perf_counter() - start_time) * 1_000
    mode = result["mode"]
    threshold = threshold_for(model, mode)
    is_fraud = probability >= threshold
    logger.debug("DeFiGuard mode=%s p=%.4f threshold=%.3f is_fraud=%s latency=%.2f ms",
                 mode, probability, threshold, is_fraud, latency_ms)

    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_probability=round(probability, 6),
        threshold=round(threshold, 6),
        mode=mode,
        exec_time_ms=round(latency_ms, 3),
    )
