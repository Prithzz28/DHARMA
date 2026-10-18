# -*- coding: utf-8 -*-
"""
inference_engine/api/dependencies.py
======================================
Global application state and FastAPI dependency injection helpers.

Both models are loaded **exactly once** at startup via the lifespan context
manager defined in ``api/main.py`` and stored in the ``AppState`` singleton
below.  All request handlers receive the loaded model(s) via
``Depends(get_model)`` / ``Depends(get_defiguard_model)`` — this avoids any
per-request file I/O overhead.

Model loading strategy
-----------------------
* The Random Forest model is **required** — the server will start in a
  degraded state (503 on /predict) if it is absent.
* The DeFiGuard model is **optional** — if the file is missing or fails to
  load (e.g. PyTorch not installed), the engine silently falls back to
  RF-only mode and logs a warning.
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
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure models package is on sys.path so DeFiGuardPredictor can be resolved
# by joblib/pickle during deserialisation of defi_guard_model.joblib.
# ---------------------------------------------------------------------------
_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent  # backend/
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


# ---------------------------------------------------------------------------
# Stub pre-loader — must run BEFORE joblib.load() for DeFiGuard
# ---------------------------------------------------------------------------

def _import_defiguard_stubs() -> None:
    """Pre-register all classes referenced in defi_guard_model.joblib pickle stream.

    joblib/pickle resolves class names via ``sys.modules`` at deserialization
    time.  All required module+class pairs must be importable before the load
    call, otherwise Python raises ``ModuleNotFoundError`` or ``AttributeError``.

    Classes needed (reverse-engineered from the pickle stream):
        models.defiguard.{TabularEncoder, GraphEncoder, CrossAttentionFusion, DeFiGuard}
        models.predictor.DeFiGuardPredictor
        torch_geometric.nn.conv.gatv2_conv.GATv2Conv
        torch_geometric.nn.aggr.basic.SumAggregation
        torch_geometric.inspector.Inspector
    """
    # models stubs
    from models.defiguard import (  # noqa: F401  (side-effect imports)
        TabularEncoder,
        GraphEncoder,
        CrossAttentionFusion,
        DeFiGuard,
    )
    from models.predictor import DeFiGuardPredictor  # noqa: F401

    # torch_geometric stubs (local compatibility package under backend/torch_geometric/)
    from torch_geometric.nn.conv.gatv2_conv import GATv2Conv  # noqa: F401
    from torch_geometric.nn.aggr.basic import SumAggregation  # noqa: F401
    from torch_geometric.inspector import Inspector, Signature, Parameter  # noqa: F401

    logger.debug("DeFiGuard stubs imported successfully.")


@dataclass
class AppState:
    """Thread-safe container for all loaded models and their metadata.

    This is a plain dataclass (not a Pydantic model) so it can hold
    non-serialisable sklearn / PyTorch objects.
    """

    # ----- Random Forest (required) ----- #
    model: Optional[RandomForestClassifier] = field(default=None)
    model_path: str = field(default="")
    is_loaded: bool = field(default=False)

    # ----- DeFiGuard (optional) ----- #
    defiguard_model: Optional[Any] = field(default=None)
    defiguard_model_path: str = field(default="")
    defiguard_loaded: bool = field(default=False)

    # ----- Live telemetry counters ----- #
    total_predictions: int = field(default=0)
    total_fraud: int = field(default=0)
    sum_latency_ms: float = field(default=0.0)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def record_prediction(self, is_fraud: bool, latency_ms: float) -> None:
        """Thread-safe-enough counter update (GIL protects simple int increments)."""
        self.total_predictions += 1
        if is_fraud:
            self.total_fraud += 1
        self.sum_latency_ms += latency_ms

    def load(self, model_path: Path) -> None:
        """Load (or reload) the Random Forest model from *model_path*."""
        backend_models_dir = Path(__file__).resolve().parent.parent.parent / "models"
        resolved_path = self._resolve_path(
            model_path,
            candidates=[
                Path("./models/random_forest_fraud_model.joblib"),
                Path("./backend/models/random_forest_fraud_model.joblib"),
                backend_models_dir / "random_forest_fraud_model.joblib",
                Path("../models/random_forest_fraud_model.joblib"),
            ],
        )
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Random Forest model not found: {model_path}\n"
                "Run the training pipeline first or set the MODEL_PATH env variable."
            )
        logger.info("Loading Random Forest model from %s …", resolved_path)
        self.model = joblib.load(resolved_path)
        self.model_path = str(resolved_path.resolve())
        self.is_loaded = True
        logger.info(
            "✅ Random Forest loaded — estimators: %d, features: %d",
            self.model.n_estimators,
            self.model.n_features_in_,
        )

    def load_defiguard(self, model_path: Path) -> None:
        """Load (or reload) the DeFiGuard model from *model_path*.

        Failure is **non-fatal** — the engine continues in RF-only mode.
        """
        backend_models_dir = Path(__file__).resolve().parent.parent.parent / "models"
        resolved_path = self._resolve_path(
            model_path,
            candidates=[
                Path("./models/defi_guard_model.joblib"),
                Path("./backend/models/defi_guard_model.joblib"),
                backend_models_dir / "defi_guard_model.joblib",
                Path("../models/defi_guard_model.joblib"),
            ],
        )
        if not resolved_path.exists():
            logger.warning(
                "⚠️  DeFiGuard model not found at %s — running RF-only mode.", model_path
            )
            return

        # Import the stub classes BEFORE joblib.load so pickle can resolve them
        try:
            _import_defiguard_stubs()
        except ImportError as exc:
            logger.warning("⚠️  DeFiGuard class stubs unavailable (%s) — RF-only mode.", exc)
            return

        try:
            logger.info("Loading DeFiGuard model from %s …", resolved_path)
            self.defiguard_model = joblib.load(resolved_path)
            self.defiguard_model_path = str(resolved_path.resolve())
            self.defiguard_loaded = True
            logger.info(
                "✅ DeFiGuard loaded — type: %s", type(self.defiguard_model).__name__
            )
        except Exception as exc:
            logger.warning(
                "⚠️  Failed to load DeFiGuard model (%s: %s) — RF-only mode.",
                type(exc).__name__,
                exc,
            )

    @staticmethod
    def _resolve_path(primary: Path, candidates: list[Path]) -> Path:
        """Return *primary* if it exists, else the first existing candidate."""
        if primary.exists():
            return primary
        for c in candidates:
            if c.exists():
                return c
        return primary  # will fail later with a clear FileNotFoundError


# ---------------------------------------------------------------------------
# Module-level singleton shared across all requests.
# ---------------------------------------------------------------------------
_app_state = AppState()
_model_state = _app_state  # backward-compatibility alias for tests and legacy callers


def get_app_state() -> AppState:
    """Return the shared AppState instance (used in lifespan)."""
    return _app_state


# ---------------------------------------------------------------------------
# FastAPI dependency helpers
# ---------------------------------------------------------------------------

def get_model() -> RandomForestClassifier:
    """FastAPI dependency: return the loaded RF model or raise 503."""
    if not _app_state.is_loaded or _app_state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Random Forest model not yet loaded. The service is still starting up.",
        )
    return _app_state.model


def get_defiguard_model() -> Optional[Any]:
    """FastAPI dependency: return the DeFiGuard model, or None if unavailable."""
    return _app_state.defiguard_model if _app_state.defiguard_loaded else None


# ---------------------------------------------------------------------------
# Legacy alias — keep backward compatibility with code that imports get_model_state
# ---------------------------------------------------------------------------
def get_model_state() -> AppState:
    """Backward-compatible alias for ``get_app_state()``."""
    return _app_state
