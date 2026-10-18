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
schema (and ``data_loader.TRANSACTION_FEATURES``).  All values are Python
``float``.

Error handling
--------------
- ``ExtractionError`` is raised only for *completely malformed* payloads (e.g.
  not a dict).
- Individual missing fields log a warning and fall back to 0.0 so the pipeline
  is never dropped for a single absent field.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from feature_extractor.metrics import (
    addr_length,
    calc_event_activity_flag,
    calc_gas_efficiency,
    calc_gas_per_log_event,
    calc_gas_price_ratio,
    calc_is_same_address,
    calc_length_log,
    calc_log_removed,
    calc_normalized_token_transfer,
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
            return float(int(value, 16))
        return float(value)
    except (ValueError, TypeError):
        logger.warning("Could not convert field '%s' value %r to float — defaulting to %s",
                       field_name, value, default)
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Parse hex or int to int; return default on failure."""
    if value is None:
        return default
    try:
        if isinstance(value, str) and value.startswith("0x"):
            return int(value, 16)
        return int(value)
    except (ValueError, TypeError):
        return default


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
    tx_hash: Optional[str] = tx_payload.get("hash") or tx_payload.get("transactionHash")
    from_addr: Optional[str] = tx_payload.get("from")
    to_addr: Optional[str] = tx_payload.get("to")

    # Gas limit (sent by user) vs gas used (confirmed in receipt)
    gas_limit = _hex_to_float(tx_payload.get("gas"), "gas", default=21_000.0)
    # For pending transactions gasUsed is absent; estimate as 70% of gas_limit
    # rather than assuming 100% usage (which makes gas_efficiency always 1.0).
    raw_gas_used = tx_payload.get("gasUsed")
    if raw_gas_used is not None:
        gas_used = _hex_to_float(raw_gas_used, "gasUsed", default=gas_limit)
    else:
        gas_used = gas_limit * 0.7

    # EIP-1559 fields with legacy fallback
    effective_gas_price = _hex_to_float(
        tx_payload.get("effectiveGasPrice")
        or tx_payload.get("gasPrice")
        or tx_payload.get("maxFeePerGas"),
        "effectiveGasPrice",
        default=0.0,
    )
    base_fee_per_gas: Optional[float] = (
        _hex_to_float(tx_payload["baseFeePerGas"], "baseFeePerGas")
        if "baseFeePerGas" in tx_payload
        else None
    )

    value = _hex_to_float(tx_payload.get("value", "0x0"), "value", default=0.0)
    chain_id = _hex_to_float(tx_payload.get("chainId", "0x1"), "chainId", default=1.0)
    block_number = _hex_to_float(tx_payload.get("blockNumber", "0x0"), "blockNumber", default=0.0)
    tx_index = _hex_to_float(tx_payload.get("transactionIndex", "0x0"), "transactionIndex", default=0.0)
    cumulative_gas_used = _hex_to_float(
        tx_payload.get("cumulativeGasUsed", "0x0"), "cumulativeGasUsed", default=0.0
    )

    # Logs — present in receipts; empty list for pending transactions
    logs: list = tx_payload.get("logs", [])
    log_count: int = len(logs)

    # ------------------------------------------------------------------ #
    # Derived metrics
    # ------------------------------------------------------------------ #
    total_gas_cost = calc_total_gas_cost(gas_used, effective_gas_price)
    gas_efficiency = calc_gas_efficiency(gas_used, gas_limit)
    gas_price_ratio = calc_gas_price_ratio(effective_gas_price, base_fee_per_gas)
    gas_per_log_event = calc_gas_per_log_event(gas_used, log_count)

    length_log = calc_length_log(log_count)
    event_activity_flag = calc_event_activity_flag(log_count)
    log_removed = calc_log_removed(logs)

    is_same_address = calc_is_same_address(from_addr or "", to_addr)
    normalized_token_transfer = calc_normalized_token_transfer(value)

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
        "gas_per_log_event": gas_per_log_event,
        "event_activity_flag": event_activity_flag,
        "normalized_token_transfer": normalized_token_transfer,
        "effective_gas_price": effective_gas_price,
        "cumulative_gas_used": cumulative_gas_used,
        "is_same_address": is_same_address,
        "gas_price_ratio": gas_price_ratio,
        "length_log": length_log,
        "log_count": float(log_count),
    }

    logger.debug("Extracted features: %s", features)
    return features
