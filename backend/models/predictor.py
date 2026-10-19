# -*- coding: utf-8 -*-
"""
models/predictor.py
====================
``DeFiGuardPredictor`` — the wrapper object stored in ``defi_guard_model.joblib``
— plus the inductive ego-graph builder used at inference time.

The graph builder and forward pass are ported from the DeFiGuard training
repository (``models/predictor.py``) so that inference matches training:

    tabular (45 features, StandardScaler)  ─┐
                                             ├─ DeFiGuard.forward → sigmoid
    1-hop ego-graph (tx + address nodes)    ─┘

The pickled object carries ``model_state_dict``, ``scaler_mean``,
``scaler_scale``, ``model_config``, ``feature_names`` and ``training_metrics``.
On first use the network is rebuilt from ``model_state_dict`` with a *strict*
``load_state_dict`` (any architecture mismatch fails loudly) and memoised.

The class path ``models.predictor.DeFiGuardPredictor`` is referenced by the
pickle stream — do NOT rename or move it.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch

from models.defiguard import DeFiGuard

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants (must match the training repository)
# --------------------------------------------------------------------------- #
TX_NODE_DIM = 8
ADDR_NODE_DIM = 4
EDGE_FEATURE_DIM = 2
MAX_NODE_DIM = max(TX_NODE_DIM, ADDR_NODE_DIM)

DEFAULT_TABULAR_FEATURES: List[str] = [
    "block_number", "chain_id", "length_to", "length_log",
    "cumulative_gas_used", "effective_gas_price", "total_gas_cost",
    "gas_used", "gas_price_ratio", "is_contract_creation",
    "is_same_address", "gas_efficiency", "value", "duration_seconds",
    "error_rate", "number_of_errors",
    "log_removed", "status", "index", "gas_per_log_event",
    "event_activity_flag", "normalized_token_transfer",
    "token_transfer_amount", "log_count", "num_transaction",
    "gas_used.gas_used_average", "gas_used.gas_used_standard_deviation",
    "gas_used.gas_used_coefficient_of_variation",
    "gas_prices.gas_prices_average", "gas_prices.gas_prices_standard_deviation",
    "values.values_average", "values.values_standard_deviation",
    "nonce.nonce_average",
    "erc_20_Name_Length", "erc_20_Symbol_Length",
    "erc_20_Name_Uppercase_Ratio", "erc_20_Symbol_Is_Uppercase",
    "erc_20_Name_Entropy", "erc_20_Symbol_Entropy",
    "erc_20_TokenDivisor", "erc_20_Normalized_Quantity",
    "erc_721_Name_Length", "erc_721_Symbol_Length",
    "erc_721_Name_Entropy", "erc_721_Symbol_Entropy",
]


# --------------------------------------------------------------------------- #
# Parsing helpers (identical semantics to the training repository)
# --------------------------------------------------------------------------- #
def _safe_float(val: Any, default: float = 0.0) -> float:
    """Parse float / bool / status string to float; NaN and inf become *default*."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        if np.isnan(val) or np.isinf(val):
            return default
        return float(val)
    val_str = str(val).strip().lower()
    if val_str in ("1", "true", "ok", "success", "0x1"):
        return 1.0
    if val_str in ("0", "false", "notok", "failed", "0x0"):
        return 0.0
    try:
        f = float(val)
    except (ValueError, TypeError):
        return default
    if np.isnan(f) or np.isinf(f):
        return default
    return f


def _safe_log(val: Any, default: float = 0.0) -> float:
    """log1p of a non-negative value; *default* otherwise."""
    f = _safe_float(val, default=0.0)
    return float(np.log1p(f)) if f > 0 else default


def _hash_to_int(addr_str: Any, bits: int = 16) -> float:
    """Stable [0, 1] encoding of an address string."""
    if not addr_str:
        return 0.0
    h = hashlib.md5(str(addr_str).lower().encode()).hexdigest()
    return float(int(h[:4], 16) / (2**bits - 1))


def _addr_node(addr: str, role: float, kind: float) -> np.ndarray:
    feat = np.zeros(ADDR_NODE_DIM, dtype=np.float32)
    feat[0] = len(addr)
    feat[1] = role
    feat[2] = _hash_to_int(addr)
    feat[3] = kind
    return feat


def build_ego_graph_for_inference(tx_data: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """Construct the inductive 1-hop ego-graph for a single transaction.

    Nodes: 0 = transaction (type 0); sender, recipient, contract and log-emitting
    addresses (type 1). Edges: FROM→TX, TX→TO, FROM→TO, TX→CONTRACT/LOG.
    """
    nodes: List[tuple] = []
    edges: List[tuple] = []
    addr_to_idx: Dict[str, int] = {}

    tx_features = np.zeros(TX_NODE_DIM, dtype=np.float32)
    tx_features[0] = _safe_log(tx_data.get("gas_used", 0))
    tx_features[1] = _safe_log(tx_data.get("value", 0))
    tx_features[2] = _safe_float(tx_data.get("status", 0))
    tx_features[3] = _safe_log(tx_data.get("effective_gas_price", tx_data.get("gas_price", 0)))
    tx_features[4] = _safe_log(tx_data.get("cumulative_gas_used", 0))
    tx_features[5] = _safe_log(tx_data.get("block_number", 0))
    tx_features[6] = _safe_float(tx_data.get("chain_id", 1), default=1.0)
    tx_features[7] = _safe_log(tx_data.get("nonce", 0))
    nodes.append((0, tx_features))

    transfer_edge = np.array(
        [_safe_log(tx_data.get("gas_used", 0)), _safe_log(tx_data.get("value", 0))],
        dtype=np.float32,
    )

    from_addr = str(tx_data.get("from_address", tx_data.get("from", "")) or "").lower()
    if from_addr:
        from_idx = len(nodes)
        addr_to_idx[from_addr] = from_idx
        nodes.append((1, _addr_node(from_addr, 0.0, 0.0)))
        edges.append((from_idx, 0, transfer_edge.copy()))

    to_addr = str(tx_data.get("to_address", tx_data.get("to", "")) or "").lower()
    if to_addr and to_addr != from_addr:
        to_idx = len(nodes)
        addr_to_idx[to_addr] = to_idx
        nodes.append((1, _addr_node(to_addr, 1.0, 1.0)))
        edges.append((0, to_idx, transfer_edge.copy()))
        if from_addr in addr_to_idx:
            edges.append((addr_to_idx[from_addr], to_idx, transfer_edge.copy()))

    contract_addr = str(tx_data.get("contract_address", "") or "").lower()
    if contract_addr and contract_addr not in addr_to_idx:
        c_idx = len(nodes)
        addr_to_idx[contract_addr] = c_idx
        nodes.append((1, _addr_node(contract_addr, 1.0, 2.0)))
        edges.append((0, c_idx, np.zeros(EDGE_FEATURE_DIM, dtype=np.float32)))

    logs = tx_data.get("logs")
    for log_event in logs if isinstance(logs, list) else []:
        if not isinstance(log_event, dict):
            continue
        log_addr = str(log_event.get("address", "") or "").lower()
        if log_addr and log_addr not in addr_to_idx:
            l_idx = len(nodes)
            addr_to_idx[log_addr] = l_idx
            nodes.append((1, _addr_node(log_addr, 1.0, 3.0)))
            edges.append((0, l_idx, np.zeros(EDGE_FEATURE_DIM, dtype=np.float32)))

    node_features = np.zeros((len(nodes), MAX_NODE_DIM), dtype=np.float32)
    node_types = np.zeros(len(nodes), dtype=np.int64)
    for i, (ntype, feat) in enumerate(nodes):
        node_types[i] = ntype
        node_features[i, : len(feat)] = feat

    if edges:
        edge_index = np.array([[e[0], e[1]] for e in edges], dtype=np.int64).T
        edge_features = np.stack([e[2] for e in edges])
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_features = np.zeros((0, EDGE_FEATURE_DIM), dtype=np.float32)

    return {
        "node_features": node_features,
        "node_types": node_types,
        "edge_index": edge_index,
        "edge_features": edge_features,
    }


# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #
class DeFiGuardPredictor:
    """Inference wrapper around the full multi-modal DeFiGuard network."""

    def __init__(
        self,
        model_state_dict: Dict[str, Any],
        scaler_mean: Any,
        scaler_scale: Any,
        model_config: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
        training_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_state_dict = model_state_dict
        self.scaler_mean = np.asarray(scaler_mean, dtype=np.float32)
        self.scaler_scale = np.asarray(scaler_scale, dtype=np.float32)
        self.model_config = model_config or {
            "tabular_dim": 45, "node_feature_dim": 8, "edge_feature_dim": 2,
            "hidden_dim": 128, "embed_dim": 64, "n_heads": 4, "n_node_types": 3,
        }
        self.feature_names = list(feature_names or DEFAULT_TABULAR_FEATURES)
        self.training_metrics = training_metrics or {}
        self.classes_ = np.array([0, 1])
        self._model: Optional[DeFiGuard] = None
        self._lock = threading.Lock()

    # The exported pickle also contains a pre-built ``_model``; it is discarded
    # so the network is always rebuilt from ``model_state_dict`` (strictly).
    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_model"] = None
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._model = None
        self._lock = threading.Lock()

    def get_model(self) -> DeFiGuard:
        """Build (once) the DeFiGuard network from the stored weights, in eval mode."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    cfg = self.model_config
                    model = DeFiGuard(
                        tabular_dim=cfg["tabular_dim"],
                        node_feature_dim=cfg["node_feature_dim"],
                        edge_feature_dim=cfg["edge_feature_dim"],
                        hidden_dim=cfg["hidden_dim"],
                        embed_dim=cfg["embed_dim"],
                        n_heads=cfg["n_heads"],
                        n_node_types=cfg["n_node_types"],
                        dropout=0.0,
                    )
                    model.load_state_dict(self.model_state_dict, strict=True)
                    model.eval()
                    self._model = model
        return self._model

    def _scaled_tabular(self, tx_data: Mapping[str, Any]) -> np.ndarray:
        # Values beyond float32 range overflow to inf (the reference predictor
        # then returns NaN); they are mapped to 0 here, like the training
        # pipeline's inf → 0 cleaning.
        with np.errstate(over="ignore", invalid="ignore"):
            raw = np.array(
                [_safe_float(tx_data.get(name, 0.0)) for name in self.feature_names],
                dtype=np.float32,
            )
            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
            # Guard zero-variance columns (sklearn would store scale 1).
            scale = np.where(self.scaler_scale == 0, 1.0, self.scaler_scale).astype(np.float32)
            scaled = (raw - self.scaler_mean) / scale
        return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def predict_fraud_proba(
        self,
        features: Mapping[str, Any],
        raw_tx: Optional[Mapping[str, Any]] = None,
    ) -> float:
        """Fraud probability in [0, 1] for one transaction.

        Parameters
        ----------
        features:
            Feature dict (the 20 API features; missing DeFiGuard features are 0).
        raw_tx:
            Optional raw transaction fields used for the ego-graph — ``from``,
            ``to``, ``nonce``, ``status``, ``logs``, ``contract_address``. Without
            it the graph has only the transaction node.
        """
        tx_data: Dict[str, Any] = dict(features)
        if raw_tx:
            for key in ("from", "to", "nonce", "status", "logs", "contract_address"):
                if raw_tx.get(key) is not None:
                    tx_data[key] = raw_tx[key]

        model = self.get_model()
        tabular = torch.from_numpy(self._scaled_tabular(tx_data)).unsqueeze(0)
        graph = build_ego_graph_for_inference(tx_data)
        node_features = torch.from_numpy(graph["node_features"])
        with torch.no_grad():
            logits, _ = model(
                tabular,
                node_features,
                torch.from_numpy(graph["node_types"]),
                torch.from_numpy(graph["edge_index"]),
                torch.from_numpy(graph["edge_features"]),
                torch.zeros(node_features.shape[0], dtype=torch.long),
            )
            prob = float(torch.sigmoid(logits).item())
        if not np.isfinite(prob):
            raise ValueError("DeFiGuard produced a non-finite probability.")
        return prob
