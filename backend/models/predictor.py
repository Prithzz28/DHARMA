"""
predictor.py — DeFiGuard Predictor for Web3 Firewalls

``DeFiGuardPredictor`` is the object stored in ``defi_guard_model.joblib``. It
scores ONE raw Ethereum transaction with the exact feature/graph code used in
training (models/features.py), in one of two modes:

  receipt  — transaction + execution receipt (mined, or simulated before
             signing): full DeFiGuard (tabular + receipt ego-graph, cross-attention)
  pending  — signed/unsigned transaction only (wallet / mempool): pre-execution
             MLP over PRE_EXEC_FEATURES; no receipt graph exists yet

Each mode carries the decision threshold selected on address-disjoint
validation data during training.

The class path ``models.predictor.DeFiGuardPredictor`` is referenced by the
pickle stream — do NOT rename or move it. This file is kept byte-identical in
the DeFiGuard research repository and the Mini Pro firewall backend.
"""

import threading

import numpy as np
import torch

from models.defiguard import DeFiGuard, TabularOnlyModel
from models.features import (
    ALL_FEATURES, PRE_EXEC_FEATURES, split_rpc_result, tabular_vector, build_ego_graph,
)


class DeFiGuardPredictor:
    """Inference wrapper holding both operating modes."""

    def __init__(self, receipt_state_dict, pending_state_dict, scaler_mean, scaler_scale,
                 model_config, thresholds, metrics=None, metadata=None):
        self.receipt_state_dict = receipt_state_dict
        self.pending_state_dict = pending_state_dict
        self.scaler_mean = np.asarray(scaler_mean, dtype=np.float32)
        self.scaler_scale = np.asarray(scaler_scale, dtype=np.float32)
        self.model_config = dict(model_config)
        self.thresholds = dict(thresholds)
        self.metrics = metrics or {}
        self.metadata = metadata or {}
        self.feature_names = list(ALL_FEATURES)
        self.pending_feature_names = list(PRE_EXEC_FEATURES)
        self._pending_idx = np.array([ALL_FEATURES.index(f) for f in PRE_EXEC_FEATURES])
        self._models = None
        self._lock = threading.Lock()

    # Rebuild the networks lazily from the state dicts after unpickling.
    def __getstate__(self):
        state = dict(self.__dict__)
        state["_models"] = None
        state.pop("_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._models = None
        self._lock = threading.Lock()

    def _get_models(self):
        if self._models is None:
            with self._lock:
                if self._models is None:
                    cfg = self.model_config
                    receipt = DeFiGuard(
                        tabular_dim=len(ALL_FEATURES),
                        node_feature_dim=cfg["node_feature_dim"],
                        edge_feature_dim=cfg["edge_feature_dim"],
                        hidden_dim=cfg["hidden_dim"],
                        embed_dim=cfg["embed_dim"],
                        n_heads=cfg["n_heads"],
                        n_node_types=cfg["n_node_types"],
                        dropout=0.0,
                    )
                    receipt.load_state_dict(self.receipt_state_dict, strict=True)
                    pending = TabularOnlyModel(len(PRE_EXEC_FEATURES), cfg["hidden_dim"], 0.0)
                    pending.load_state_dict(self.pending_state_dict, strict=True)
                    self._models = {"receipt": receipt.eval(), "pending": pending.eval()}
        return self._models

    def _scale(self, raw):
        return ((raw - self.scaler_mean) / self.scaler_scale).astype(np.float32)

    @torch.no_grad()
    def score(self, tx, receipt=None):
        """Score one transaction.

        Args:
            tx: JSON-RPC transaction object (eth_getTransactionByHash /
                eth_sendTransaction params): from, to, value, gas, gasPrice or
                maxFeePerGas, nonce, input/data.
            receipt: optional JSON-RPC receipt (gasUsed, status, logs, ...).
        Returns:
            dict(fraud_probability, threshold, is_fraud, mode)
        """
        models = self._get_models()
        x = self._scale(tabular_vector(tx, receipt))
        mode = "receipt" if receipt else "pending"
        if mode == "pending":
            logits, _ = models["pending"](torch.from_numpy(x[self._pending_idx]).unsqueeze(0))
        else:
            g = build_ego_graph(tx, receipt)
            n = len(g["node_types"])
            logits, _ = models["receipt"](
                torch.from_numpy(x).unsqueeze(0),
                torch.from_numpy(g["node_features"]),
                torch.from_numpy(g["node_types"]),
                torch.from_numpy(g["edge_index"]),
                torch.from_numpy(g["edge_features"]),
                torch.zeros(n, dtype=torch.long),
            )
        prob = float(torch.sigmoid(logits).item())
        if not np.isfinite(prob):
            raise ValueError("DeFiGuard produced a non-finite probability.")
        threshold = float(self.thresholds[mode])
        return {"fraud_probability": prob, "threshold": threshold,
                "is_fraud": prob >= threshold, "mode": mode}

    def score_rpc_result(self, raw):
        """Score a stored {"result": [receipt, tx]} JSON-RPC pair."""
        tx, receipt = split_rpc_result(raw)
        return self.score(tx, receipt)

    def get_model_info(self):
        return {
            "modes": {"receipt": "DeFiGuard (tabular + receipt ego-graph)",
                      "pending": "MLP over pre-execution features"},
            "thresholds": self.thresholds,
            "n_features": {"receipt": len(self.feature_names), "pending": len(self.pending_feature_names)},
            "metrics": self.metrics,
            **self.metadata,
        }

    def __repr__(self):
        return f"DeFiGuardPredictor(modes=receipt|pending, thresholds={self.thresholds})"
