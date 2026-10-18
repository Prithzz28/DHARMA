# -*- coding: utf-8 -*-
"""
feature_extractor/metrics.py
==============================
Pure-math helper functions for deriving the DeFiTransLyzer features from raw
Ethereum transaction fields.

All functions are stateless, have no external dependencies beyond the standard
library and numpy, and are optimised for single-call latency (no vectorised
batch paths — those belong in the training pipeline).

Naming follows the paper's feature glossary and ``data_loader.TRANSACTION_FEATURES``
exactly so that returned keys map 1-to-1 to the Pydantic schema without
any renaming step.
"""
from __future__ import annotations

import math
from typing import Optional


# --------------------------------------------------------------------------- #
# Gas metrics
# --------------------------------------------------------------------------- #

def calc_total_gas_cost(gas_used: float, effective_gas_price: float) -> float:
    """Total cost of the transaction in wei.

    total_gas_cost = gas_used × effective_gas_price
    """
    return gas_used * effective_gas_price


def calc_gas_efficiency(gas_used: float, gas_limit: float) -> float:
    """Ratio of gas consumed to gas limit.

    A ratio close to 1.0 suggests the sender tightly estimated gas
    (common in scripted/bot transactions).
    """
    if gas_limit <= 0:
        return 0.0
    return gas_used / gas_limit


def calc_gas_price_ratio(
    effective_gas_price: float,
    base_fee_per_gas: Optional[float],
) -> float:
    """Ratio of the effective gas price to the block base fee.

    Transactions paying many multiples of the base fee are suspicious
    (front-running / sandwich attack pattern).
    Returns 1.0 when base_fee_per_gas is unavailable (pre-EIP-1559 blocks).
    """
    if base_fee_per_gas is None or base_fee_per_gas <= 0:
        return 1.0
    return effective_gas_price / base_fee_per_gas


def calc_gas_per_log_event(gas_used: float, log_count: int) -> float:
    """Average gas spent per log event.

    gas_per_log_event = gas_used / (log_count + 1)

    Adding 1 avoids division-by-zero for transactions with no logs while
    keeping the metric meaningful (a transaction with 0 logs returns gas_used).
    """
    return gas_used / (log_count + 1)


# --------------------------------------------------------------------------- #
# Log / event metrics
# --------------------------------------------------------------------------- #

def calc_length_log(log_count: int) -> float:
    """Natural logarithm of the number of log events (+ 1 for log(0) safety).

    length_log ≈ estimated event generation complexity.
    """
    return math.log(log_count + 1)


def calc_event_activity_flag(log_count: int) -> float:
    """Binary flag: 1.0 if the transaction emitted ≥ 1 log event."""
    return 1.0 if log_count > 0 else 0.0


def calc_log_removed(logs: list) -> float:
    """1.0 if *any* log entry in the receipt has ``removed == True``.

    A removed log signals a blockchain reorganisation, which is correlated
    with certain attack patterns.
    """
    return 1.0 if any(log.get("removed", False) for log in logs) else 0.0


# --------------------------------------------------------------------------- #
# Address / structural metrics
# --------------------------------------------------------------------------- #

def calc_is_same_address(from_addr: str, to_addr: Optional[str]) -> float:
    """1.0 if the sender and recipient are the same address (self-transfer)."""
    if not from_addr or not to_addr:
        return 0.0
    return 1.0 if from_addr.lower() == to_addr.lower() else 0.0


def addr_length(address: Optional[str]) -> float:
    """Byte-length of an Ethereum address string (including '0x' prefix).

    Standard addresses are 42 characters; contracts and EOAs are
    indistinguishable by length alone, but *anomalous* lengths are a
    fraud signal.
    """
    return float(len(address)) if address else 0.0


def hash_length(tx_hash: Optional[str]) -> float:
    """Byte-length of the transaction hash string (including '0x' prefix).

    Standard hashes are 66 characters.
    """
    return float(len(tx_hash)) if tx_hash else 0.0


# --------------------------------------------------------------------------- #
# Value metrics
# --------------------------------------------------------------------------- #

def calc_normalized_token_transfer(value: float, max_value: float = 1e21) -> float:
    """Normalize the raw ETH value (wei) to [0, 1] against a reference maximum.

    The reference maximum defaults to 1000 ETH in wei, which captures the
    vast majority of DeFi transaction values without saturation.
    """
    if max_value <= 0:
        return 0.0
    return min(value / max_value, 1.0)
