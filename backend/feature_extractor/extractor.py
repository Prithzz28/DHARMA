# -*- coding: utf-8 -*-
"""
feature_extractor/extractor.py
================================
Real-time DeFiTransLyzer implementation.

Converts a raw, unconfirmed Ethereum transaction payload (standard JSON RPC
format) into the flat 20-feature dict required by the inference engine.

Supported input formats
------------------------
1. **eth_sendTransaction** — a *pending* transaction object (from MetaMask).
   Fields: ``from``, ``to``, ``gas``, ``gasPrice``, ``value``, ``data``, ``nonce``.
   Confirmed receipt fields (``gasUsed``, ``logs``, etc.) are absent and default
   to sensible values.

2. **Transaction receipt / confirmed tx** — a receipt-style object that includes
   ``gasUsed``, ``logs``, ``blockNumber``, ``transactionIndex``, etc.

3. **Mixed** — any combination of the two; the extractor performs a best-effort
   extraction of whatever is available.

Output
------
A flat ``dict`` whose keys map 1-to-1 to the Pydantic ``TransactionFeatures``
schema. All values are Python ``float``.

Feature *definitions* follow the DeFiTransLyzer training data the models were
fitted on (see ``metrics.py``), so extracted vectors stay inside the training
distribution.

Error handling
--------------
- ``ExtractionError`` is raised only for *completely malformed* payloads (e.g.
  not a dict).
- Individual missing fields log a warning and fall back to 0.0 so the pipeline
  is never dropped for a single absent field.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

from feature_extractor.metrics import (
    addr_length,
    calc_event_activity_flag,
    calc_gas_efficiency,
    calc_gas_price_ratio,
    calc_is_same_address,
    calc_length_log,
    calc_log_removed,
    calc_total_gas_cost,
    hash_length,
)

logger = logging.getLogger(__name__)


class ExtractionError(ValueError):
    """Raised when the raw payload is so malformed that no features can be extracted."""


def _hex_to_float(value: Any, field_name: str, default: float = 0.0) -> float:
    """Convert a hex string (``"0x1a2b"``), int, or float to float.

    Logs a warning and returns *default* on any parse failure.
    """
    if value is None:
        return default
    try:
        if isinstance(value, str) and value.startswith("0x"):
            result = float(int(value, 16))
        else:
            result = float(value)
    except (ValueError, TypeError, OverflowError):
        result = math.nan
    if not math.isfinite(result):
        logger.warning("Could not convert field '%s' value %r to a finite float — defaulting to %s",
                       field_name, value, default)
        return default
    return result


def _optional_str(value: Any) -> Optional[str]:
    """Return *value* as a string, or None when absent/empty."""
    if value is None or value == "":
        return None
    return value if isinstance(value, str) else str(value)


def extract_features(tx_payload: Dict[str, Any]) -> Dict[str, float]:
    """Extract the 20 DeFiTransLyzer features from a raw Ethereum transaction.

    Parameters
    ----------
    tx_payload:
        Raw JSON RPC transaction or receipt object (as returned by
        ``eth_getTransactionByHash``, ``eth_getTransactionReceipt``,
        or the ``params`` field of ``eth_sendTransaction``).

    Returns
    -------
    dict
        Flat ``{feature_name: float}`` mapping ready for the inference API.

    Raises
    ------
    ExtractionError
        If *tx_payload* is not a dict.
    """
    if not isinstance(tx_payload, dict):
        raise ExtractionError(
            f"tx_payload must be a dict, got {type(tx_payload).__name__}."
        )

    if not tx_payload:
        logger.warning("Received an empty transaction payload — all features will be zero.")

    # ------------------------------------------------------------------ #
    # Raw field extraction (hex → float / int)
    # ------------------------------------------------------------------ #
    tx_hash = _optional_str(tx_payload.get("hash") or tx_payload.get("transactionHash"))
    from_addr = _optional_str(tx_payload.get("from"))
    to_addr = _optional_str(tx_payload.get("to"))

    # Gas limit (sent by user) vs gas used (confirmed in receipt)
    gas_limit = _hex_to_float(tx_payload.get("gas"), "gas", default=21_000.0)
    # Pending transactions have no gasUsed yet; estimate it as 70% of the limit.
    raw_gas_used = tx_payload.get("gasUsed")
    if raw_gas_used is not None:
        gas_used = _hex_to_float(raw_gas_used, "gasUsed", default=gas_limit)
    else:
        gas_used = gas_limit * 0.7

    # Price offered by the sender (legacy gasPrice or EIP-1559 maxFeePerGas) and
    # the price actually paid (receipts only).
    offered_gas_price = _hex_to_float(
        tx_payload.get("gasPrice") or tx_payload.get("maxFeePerGas"), "gasPrice", default=0.0
    )
    paid_gas_price = _hex_to_float(tx_payload.get("effectiveGasPrice"), "effectiveGasPrice", default=0.0)
    effective_gas_price = paid_gas_price or offered_gas_price

    value = _hex_to_float(tx_payload.get("value", "0x0"), "value", default=0.0)
    chain_id = _hex_to_float(tx_payload.get("chainId", "0x1"), "chainId", default=1.0)
    block_number = _hex_to_float(tx_payload.get("blockNumber", "0x0"), "blockNumber", default=0.0)
    tx_index = _hex_to_float(tx_payload.get("transactionIndex", "0x0"), "transactionIndex", default=0.0)
    cumulative_gas_used = _hex_to_float(
        tx_payload.get("cumulativeGasUsed", "0x0"), "cumulativeGasUsed", default=0.0
    )

    # Logs — present in receipts; empty list for pending transactions.
    # Non-dict entries are ignored rather than crashing the request.
    raw_logs = tx_payload.get("logs")
    logs: list = [log for log in raw_logs if isinstance(log, dict)] if isinstance(raw_logs, list) else []
    log_count: int = len(logs)

    # ------------------------------------------------------------------ #
    # Derived metrics
    # ------------------------------------------------------------------ #
    total_gas_cost = calc_total_gas_cost(gas_used, effective_gas_price)
    gas_efficiency = calc_gas_efficiency(cumulative_gas_used, gas_used)
    gas_price_ratio = calc_gas_price_ratio(paid_gas_price, offered_gas_price)

    length_log = calc_length_log(log_count)
    event_activity_flag = calc_event_activity_flag(log_count)
    log_removed = calc_log_removed(logs)

    is_same_address = calc_is_same_address(from_addr or "", to_addr)

    # ------------------------------------------------------------------ #
    # Assemble feature dict (key order matches TRANSACTION_FEATURES)
    # ------------------------------------------------------------------ #
    features: Dict[str, float] = {
        "length_transaction_hash": hash_length(tx_hash),
        "length_to": addr_length(to_addr),
        "log_removed": log_removed,
        "block_number": block_number,
        "gas_used": gas_used,
        "length_from": addr_length(from_addr),
        "index": tx_index,
        "gas_efficiency": gas_efficiency,
        "value": value,
        "chain_id": chain_id,
        "total_gas_cost": total_gas_cost,
        # Constant 0 throughout the training data (the dataset never populated
        # these columns); any other value is out of distribution for the models.
        "gas_per_log_event": 0.0,
        "event_activity_flag": event_activity_flag,
        "normalized_token_transfer": 0.0,
        "effective_gas_price": effective_gas_price,
        "cumulative_gas_used": cumulative_gas_used,
        "is_same_address": is_same_address,
        "gas_price_ratio": gas_price_ratio,
        "length_log": length_log,
        "log_count": 0.0,  # constant 0 in training; the log count lives in length_log
    }

    logger.debug("Extracted features: %s", features)
    return features
