# -*- coding: utf-8 -*-
"""
inference_engine/services/ensemble.py
=======================================
Dual-model ensemble scorer for the Web3 Transaction Firewall.

Strategy
--------
Both models are run in parallel (via asyncio) and their fraud probabilities
are combined to produce the **best** (most conservative / highest-recall)
final classification:

    final_score = max(rf_score, defiguard_score)

Rationale for ``max``
~~~~~~~~~~~~~~~~~~~~~
In a fraud-prevention context, a false negative (missed fraud) is far more
costly than a false positive (flagging a legitimate transaction). Using the
maximum of two independent classifiers therefore:
  * Increases recall / sensitivity at the expense of a small precision drop.
  * Leverages model diversity: Random Forest excels at tabular patterns;
    DeFiGuard captures non-linear and graph-level relationships.
  * Degrades gracefully — if DeFiGuard is unavailable, the RF score is used.

Ensemble modes
~~~~~~~~~~~~~~
``EnsembleMode.MAX``    — final = max(rf, dg)          ← default (highest recall)
``EnsembleMode.MEAN``   — final = mean(rf, dg)          (balanced)
``EnsembleMode.RF_ONLY``— final = rf only               (fallback)
``EnsembleMode.DG_ONLY``— final = dg only               (not recommended alone)
``EnsembleMode.WEIGHTED``— final = w_rf*rf + w_dg*dg    (configurable weights)

The active mode is selected via the ``ENSEMBLE_MODE`` environment variable
(defaults to ``max``).
"""
from __future__ import annotations

import asyncio
import enum
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Reuse a single thread pool for both models
_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="ensemble_inference")

# Cache pandas availability at import time — avoids a try/import on every request.
try:
    import pandas as _pd
    _HAS_PANDAS = True
except ImportError:
    _pd = None  # type: ignore[assignment]
    _HAS_PANDAS = False


class EnsembleMode(str, enum.Enum):
    MAX = "max"
    MEAN = "mean"
    WEIGHTED = "weighted"
    RF_ONLY = "rf_only"
    DG_ONLY = "dg_only"


# ---------------------------------------------------------------------------
# Random Forest inference helpers
# ---------------------------------------------------------------------------

def _rf_feature_array(
    features: Dict[str, float],
    model: object,
) -> Any:
    """Build the feature matrix expected by the Random Forest model.

    Returns a pandas DataFrame (preferred — eliminates sklearn feature-name
    warning) if pandas is available, otherwise falls back to a plain numpy
    array. The feature-name warning suppression is handled in _sync_rf_predict.
    """
    from inference_engine.services.prediction import FEATURE_ORDER

    raw_cols = getattr(model, "feature_names_in_", None)
    cols = list(raw_cols) if raw_cols is not None else list(FEATURE_ORDER)
    values = []
    for col in cols:
        val = features.get(col, 0.0)
        try:
            values.append(float(val) if val is not None else 0.0)
        except (ValueError, TypeError):
            values.append(0.0)
    if _HAS_PANDAS:
        return _pd.DataFrame([values], columns=cols)  # type: ignore[union-attr]
    return np.array(values, dtype=np.float32).reshape(1, -1)


def _sync_rf_predict(model: object, features: Dict[str, float]) -> float:
    """Blocking RF predict_proba call — run inside executor."""
    import warnings
    arr = _rf_feature_array(features, model)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
        )
        raw = model.predict_proba(arr)  # type: ignore[union-attr]
    proba = np.asarray(raw)
    if proba.ndim == 2 and proba.shape[1] >= 2:
        return float(proba[0, 1])
    elif proba.ndim == 1 and proba.shape[0] >= 2:
        return float(proba[1])
    return float(proba.ravel()[-1])



# ---------------------------------------------------------------------------
# DeFiGuard inference helpers
# ---------------------------------------------------------------------------

def _sync_dg_predict(model: object, features: Dict[str, float]) -> float:
    """Blocking DeFiGuard predict call — run inside executor."""
    # DeFiGuardPredictor exposes predict_tabular_proba / predict_proba
    if hasattr(model, "predict_tabular_proba"):
        return float(model.predict_tabular_proba(features))  # type: ignore[union-attr]
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(features))  # type: ignore[union-attr]
    raise TypeError(f"DeFiGuard model {type(model).__name__!r} has no recognised predict method.")


# ---------------------------------------------------------------------------
# Public async ensemble API
# ---------------------------------------------------------------------------

async def ensemble_predict(
    rf_model: object,
    dg_model: Optional[object],
    features: Dict[str, float],
    mode: EnsembleMode = EnsembleMode.MAX,
    rf_weight: float = 0.45,
    dg_weight: float = 0.55,
) -> Tuple[float, float, Optional[float], str]:
    """Run both models concurrently and return the ensemble fraud probability.

    Parameters
    ----------
    rf_model:
        Loaded ``RandomForestClassifier`` instance.
    dg_model:
        Optional loaded ``DeFiGuardPredictor`` instance. Pass ``None`` to
        automatically use ``EnsembleMode.RF_ONLY``.
    features:
        Flat ``{feature_name: value}`` dict from the feature extractor.
    mode:
        Combination strategy (default ``MAX``).
    rf_weight:
        Weight for RF score in ``WEIGHTED`` mode (default 0.45).
    dg_weight:
        Weight for DG score in ``WEIGHTED`` mode (default 0.55).

    Returns
    -------
    (final_probability, rf_score, dg_score, active_mode)
        * ``final_probability`` — combined fraud score in [0, 1].
        * ``rf_score``          — Random Forest raw score.
        * ``dg_score``          — DeFiGuard raw score (None if not loaded).
        * ``active_mode``       — which mode was actually applied.
    """
    if dg_model is None:
        active_mode = EnsembleMode.RF_ONLY
    else:
        active_mode = mode

    loop = asyncio.get_running_loop()

    rf_task = loop.run_in_executor(_executor, _sync_rf_predict, rf_model, features)

    dg_score: Optional[float] = None
    if active_mode != EnsembleMode.RF_ONLY:
        dg_task = loop.run_in_executor(_executor, _sync_dg_predict, dg_model, features)
        results = await asyncio.gather(rf_task, dg_task, return_exceptions=True)
        rf_res, dg_res = results[0], results[1]

        if isinstance(rf_res, Exception):
            raise rf_res
        rf_score = rf_res

        if isinstance(dg_res, Exception):
            logger.warning(
                "DeFiGuard inference failed — falling back to RF only. Reason: %s", dg_res
            )
            dg_score = None
            active_mode = EnsembleMode.RF_ONLY
        else:
            dg_score = dg_res
    else:
        rf_score = await rf_task

    # Clean any potential NaN or Inf in scores
    if dg_score is not None and (np.isnan(dg_score) or np.isinf(dg_score)):
        dg_score = 0.5
    if np.isnan(rf_score) or np.isinf(rf_score):
        rf_score = 0.5

    # Combine scores
    if active_mode == EnsembleMode.RF_ONLY or dg_score is None:
        final = rf_score
        active_mode_str = "rf_only"
    elif active_mode == EnsembleMode.DG_ONLY:
        final = dg_score
        active_mode_str = "dg_only"
    elif active_mode == EnsembleMode.MEAN:
        final = (rf_score + dg_score) / 2.0
        active_mode_str = "mean"
    elif active_mode == EnsembleMode.WEIGHTED:
        total_weight = rf_weight + dg_weight
        if total_weight <= 0:
            total_weight = 1.0
        final = (rf_weight * rf_score + dg_weight * dg_score) / total_weight
        active_mode_str = "weighted"
    else:  # MAX (default)
        final = max(rf_score, dg_score)
        active_mode_str = "max"

    # Ensure final score is valid bounded float
    if np.isnan(final) or np.isinf(final):
        final = rf_score if not np.isnan(rf_score) else 0.5
    final = max(0.0, min(1.0, float(final)))

    logger.debug(
        "Ensemble [%s] rf=%.4f dg=%s final=%.4f",
        active_mode_str,
        rf_score,
        f"{dg_score:.4f}" if dg_score is not None else "N/A",
        final,
    )

    return final, rf_score, dg_score, active_mode_str
