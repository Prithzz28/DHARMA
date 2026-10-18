# -*- coding: utf-8 -*-
"""
models/predictor.py
====================
Class definitions required for deserialising and running ``defi_guard_model.joblib``.

The DeFiGuard model was serialised via ``joblib.dump()`` with pickle, which
stores objects by *class path*. At deserialization time Python must find the
class ``models.predictor.DeFiGuardPredictor`` in the import path.

When deserialized, ``DeFiGuardPredictor`` contains:
    - ``_model``: PyTorch ``DeFiGuard`` nn.Module (TabularEncoder, GraphEncoder, CrossAttentionFusion, Classifier)
    - ``model_state_dict``: OrderedDict of learned weights
    - ``feature_names``: List of 45 tabular feature names
    - ``scaler_mean``: NumPy array for StandardScaler mean
    - ``scaler_scale``: NumPy array for StandardScaler scale/std
    - ``training_metrics``: Dict with train/val metrics
    - ``model_config``: Dict with architecture hyperparams
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    class _FakeModule:
        def __init__(self, *a: Any, **kw: Any) -> None: ...
        def __call__(self, *a: Any, **kw: Any) -> Any: ...
    class nn:  # type: ignore[no-redef]
        Module = _FakeModule


# ---------------------------------------------------------------------------
# Default tabular feature ordering (45 features matching DeFiGuard training)
# ---------------------------------------------------------------------------
_DEFAULT_TABULAR_FEATURES: List[str] = [
    "block_number",
    "chain_id",
    "length_to",
    "length_log",
    "cumulative_gas_used",
    "effective_gas_price",
    "total_gas_cost",
    "gas_used",
    "gas_price_ratio",
    "is_contract_creation",
    "is_same_address",
    "gas_efficiency",
    "value",
    "duration_seconds",
    "error_rate",
    "number_of_errors",
    "log_removed",
    "status",
    "index",
    "gas_per_log_event",
    "event_activity_flag",
    "normalized_token_transfer",
    "token_transfer_amount",
    "log_count",
    "num_transaction",
    "gas_used.gas_used_average",
    "gas_used.gas_used_standard_deviation",
    "gas_used.gas_used_coefficient_of_variation",
    "gas_prices.gas_prices_average",
    "gas_prices.gas_prices_standard_deviation",
    "values.values_average",
    "values.values_standard_deviation",
    "nonce.nonce_average",
    "erc_20_Name_Length",
    "erc_20_Symbol_Length",
    "erc_20_Name_Uppercase_Ratio",
    "erc_20_Symbol_Is_Uppercase",
    "erc_20_Name_Entropy",
    "erc_20_Symbol_Entropy",
    "erc_20_TokenDivisor",
    "erc_20_Normalized_Quantity",
    "erc_721_Name_Length",
    "erc_721_Symbol_Length",
    "erc_721_Name_Entropy",
    "erc_721_Symbol_Entropy",
]


class DeFiGuardPredictor:
    """
    Wrapper predictor class for the DeFiGuard model deserialised from joblib.
    """

    def __init__(self) -> None:
        from models.defiguard import DeFiGuard

        self._model: Optional[DeFiGuard] = DeFiGuard() if _TORCH_AVAILABLE else None
        self.model_state_dict: Optional[Dict[str, Any]] = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_scale: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self.model_config: Optional[Dict[str, Any]] = None
        self.feature_names: List[str] = list(_DEFAULT_TABULAR_FEATURES)
        self.training_metrics: Optional[Dict[str, Any]] = None
        self.classes_: Optional[np.ndarray] = np.array([0, 1])

    def get_model(self) -> Optional[Any]:
        """Return the inner PyTorch model instance."""
        return getattr(self, "_model", None) or getattr(self, "model", None)

    def eval(self) -> DeFiGuardPredictor:
        """Set the underlying PyTorch model to evaluation mode."""
        m = self.get_model()
        if m is not None and hasattr(m, "eval"):
            m.eval()
        return self

    def train(self, mode: bool = True) -> DeFiGuardPredictor:
        """Set the underlying PyTorch model to training mode."""
        m = self.get_model()
        if m is not None and hasattr(m, "train"):
            m.train(mode)
        return self

    def predict_tabular_proba(
        self,
        features: Union[Dict[str, float], np.ndarray],
        feature_order: Optional[Sequence[str]] = None,
    ) -> float:
        """Score a transaction feature dictionary or array via the tabular path.

        Parameters
        ----------
        features:
            Dictionary mapping feature name -> float, or a NumPy array.
        feature_order:
            Optional sequence of column names. Defaults to ``self.feature_names``.

        Returns
        -------
        float
            Fraud probability in [0, 1].
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for DeFiGuard inference.")

        m = self.get_model()
        if m is None:
            raise RuntimeError("Underlying DeFiGuard model is not initialized.")

        # Prepare 2D float32 numpy array with bounds safety
        if isinstance(features, dict):
            cols = feature_order or getattr(self, "feature_names", None) or _DEFAULT_TABULAR_FEATURES
            row = []
            for col in cols:
                val = features.get(col, 0.0)
                try:
                    val = float(val) if val is not None else 0.0
                    if np.isnan(val) or np.isinf(val):
                        val = 0.0
                    val = max(-1e30, min(1e30, val))
                    row.append(val)
                except (ValueError, TypeError):
                    row.append(0.0)
            arr = np.array(row, dtype=np.float32).reshape(1, -1)
        elif isinstance(features, np.ndarray):
            arr = features.astype(np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
        else:
            raise TypeError(f"Unsupported features type: {type(features).__name__}")

        # Normalise via stored StandardScaler parameters
        scaler_mean = getattr(self, "scaler_mean", None)
        scaler_scale = getattr(self, "scaler_scale", None)
        if scaler_scale is None:
            scaler_scale = getattr(self, "scaler_std", None)

        if scaler_mean is not None and scaler_scale is not None:
            mean = np.asarray(scaler_mean, dtype=np.float32)
            scale = np.asarray(scaler_scale, dtype=np.float32)
            scale = np.where(scale == 0, 1.0, scale)
            n = min(arr.shape[1], len(mean), len(scale))
            arr[:, :n] = (arr[:, :n] - mean[:n]) / scale[:n]

        # Clean any remaining NaNs or Infs from scaling
        arr = np.nan_to_num(arr, nan=0.0, posinf=50.0, neginf=-50.0)
        tensor_x = torch.from_numpy(arr)

        m.eval()
        with torch.no_grad():
            # tabular_encoder: (batch, 45) -> (batch, 64)
            # classifier: (batch, 64) -> (batch, 1)
            tab_emb = m.tabular_encoder(tensor_x)
            logits = m.classifier(tab_emb)
            prob = torch.sigmoid(logits).squeeze().item()
            if np.isnan(prob) or np.isinf(prob):
                prob = 0.5
            prob = max(0.0, min(1.0, float(prob)))

        return float(prob)

    def predict_proba(
        self,
        features: Union[Dict[str, float], np.ndarray],
        feature_order: Optional[Sequence[str]] = None,
    ) -> Union[float, np.ndarray]:
        """Predict fraud probability. If passed a dict, returns a float in [0, 1].
        If passed a 2D numpy array, returns an array of shape (N, 2) matching sklearn.
        """
        if isinstance(features, dict):
            return self.predict_tabular_proba(features, feature_order)

        prob = self.predict_tabular_proba(features, feature_order)
        return np.array([[1.0 - prob, prob]], dtype=np.float32)

    def predict(
        self,
        features: Union[Dict[str, float], np.ndarray],
        threshold: float = 0.5,
    ) -> Union[int, np.ndarray]:
        """Predict binary class (0 or 1)."""
        if isinstance(features, dict):
            p = self.predict_tabular_proba(features)
            return int(p >= threshold)
        p = self.predict_tabular_proba(features)
        return np.array([int(p >= threshold)])
