# -*- coding: utf-8 -*-
"""
feature_extractor/extractor.py
================================
Turns a raw Ethereum JSON-RPC payload into DeFiGuard's inputs.

All feature definitions live in ``models/features.py`` (shared verbatim with
the training repository); this module only normalises the payload shapes the
API accepts into a ``(tx, receipt)`` pair.

Supported input formats
------------------------
1. **Pending transaction** — ``eth_sendTransaction`` params (from MetaMask):
   ``from``, ``to``, ``value``, ``gas``, ``gasPrice``/``maxFeePerGas``,
   ``nonce``, ``data``/``input``. Scored in *pending* mode.
2. **Transaction + receipt** — ``{"tx": {...}, "receipt": {...}}`` or a stored
   ``{"result": [receipt, tx]}`` pair. Scored in *receipt* mode.
3. **Mixed** — one object carrying transaction fields plus receipt fields
   (``gasUsed``, ``status``, ``logs``); the receipt part is split off.

Output
------
``extract_features`` returns a flat ``{feature_name: float}`` dict over
``ALL_FEATURES`` (receipt features are 0.0 in pending mode) and the mode.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from models.features import ALL_FEATURES, split_rpc_result, tabular_vector

logger = logging.getLogger(__name__)

RECEIPT_KEYS = ("gasUsed", "status", "logs", "contractAddress", "effectiveGasPrice", "cumulativeGasUsed")


class ExtractionError(ValueError):
    """Raised when the raw payload is so malformed that no features can be extracted."""


def split_payload(tx_payload: Any) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Normalise any accepted payload shape into ``(tx, receipt_or_None)``."""
    if not isinstance(tx_payload, dict):
        raise ExtractionError(f"tx_payload must be a dict, got {type(tx_payload).__name__}.")
    if "result" in tx_payload:
        tx, receipt = split_rpc_result(tx_payload)
        if tx is None:
            raise ExtractionError("'result' does not contain a transaction object.")
        return tx, receipt
    if "tx" in tx_payload:
        tx, receipt = tx_payload.get("tx"), tx_payload.get("receipt")
        if not isinstance(tx, dict) or (receipt is not None and not isinstance(receipt, dict)):
            raise ExtractionError("'tx' and 'receipt' must be JSON objects.")
        return tx, receipt or None
    if "gasUsed" in tx_payload:
        receipt = {k: tx_payload[k] for k in RECEIPT_KEYS if k in tx_payload}
        receipt.update({k: tx_payload[k] for k in ("from", "to") if k in tx_payload})
        return tx_payload, receipt
    if not tx_payload:
        logger.warning("Received an empty transaction payload — all features will be zero.")
    return tx_payload, None


def extract_features(tx_payload: Dict[str, Any]) -> Dict[str, float]:
    """Flat ``{feature: float}`` dict for any accepted payload shape."""
    tx, receipt = split_payload(tx_payload)
    try:
        vector = tabular_vector(tx, receipt)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ExtractionError(f"Could not extract features: {exc}") from exc
    return {name: float(v) for name, v in zip(ALL_FEATURES, vector)}
