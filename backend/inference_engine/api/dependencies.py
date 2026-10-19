# -*- coding: utf-8 -*-
"""
inference_engine/api/dependencies.py
======================================
Global application state and FastAPI dependency injection helpers.

The DeFiGuard model is loaded **exactly once** at startup via the lifespan
context manager in ``api/main.py`` and stored in the ``AppState`` singleton
below. Scoring endpoints receive it via ``Depends(get_defiguard_model)``,
which returns 503 while the model is unavailable.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# The DeFiGuard pickle references classes under the ``models`` package, which
# lives in backend/ — make sure it is importable however the app is launched.
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


@dataclass
class AppState:
    """Container for the loaded model, its metadata and live counters."""

    defiguard_model: Optional[Any] = None
    defiguard_model_path: str = ""
    defiguard_loaded: bool = False

    # ----- Live telemetry counters ----- #
    total_predictions: int = 0
    total_fraud: int = 0
    sum_latency_ms: float = 0.0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def record_prediction(self, is_fraud: bool, latency_ms: float) -> None:
        """Update counters. Handlers run on the single event-loop thread, so no lock is needed."""
        self.total_predictions += 1
        if is_fraud:
            self.total_fraud += 1
        self.sum_latency_ms += latency_ms

    def load_defiguard(self, model_path: Path) -> None:
        """Load (or reload) the DeFiGuard model.

        Raises FileNotFoundError if the file is missing; any other load failure
        propagates too, so startup can report it.
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"DeFiGuard model not found: {path}")
        # Importing the class makes ``models.predictor`` resolvable by pickle.
        from models.predictor import DeFiGuardPredictor

        logger.info("Loading DeFiGuard model from %s …", path)
        predictor = joblib.load(path)
        if not isinstance(predictor, DeFiGuardPredictor):
            raise TypeError(f"expected DeFiGuardPredictor, got {type(predictor).__name__}")
        predictor.get_model()  # build the network now so weight mismatches fail at startup
        self.defiguard_model = predictor
        self.defiguard_model_path = str(path.resolve())
        self.defiguard_loaded = True
        logger.info("DeFiGuard loaded (GATv2 graph encoder + cross-attention fusion).")


# ---------------------------------------------------------------------------
# Module-level singleton shared across all requests.
# ---------------------------------------------------------------------------
_app_state = AppState()


def get_app_state() -> AppState:
    """Return the shared AppState instance."""
    return _app_state


def get_defiguard_model() -> Any:
    """FastAPI dependency: return the loaded DeFiGuard model or raise 503."""
    if not _app_state.defiguard_loaded or _app_state.defiguard_model is None:
        raise HTTPException(
            status_code=503,
            detail="DeFiGuard model is not loaded. Check the server logs and DEFIGUARD_MODEL_PATH.",
        )
    return _app_state.defiguard_model
