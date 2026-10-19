# -*- coding: utf-8 -*-
"""
feature_extractor/metrics.py
==============================
Pure-math helpers deriving DeFiTransLyzer features from raw Ethereum
transaction fields.

The definitions follow the **DeFiTransLyzer dataset the models were trained
on**, as measured from its CSVs and from the StandardScaler statistics stored in
``defi_guard_model.joblib`` — not the textbook names of the columns. Where the
two differ it is noted below; feeding the textbook definition puts inputs far
outside the training distribution (e.g. a gas_price_ratio of 2 is ~200 standard
deviations from the training mean).

Features that are constant 0 throughout the training data (``log_count``,
``gas_per_log_event``, ``normalized_token_transfer``) are emitted as 0.0 by
``extractor.py`` and have no helper here.
"""
from __future__ import annotations

from typing import Optional


# --------------------------------------------------------------------------- #
# Gas metrics
# --------------------------------------------------------------------------- #

def calc_total_gas_cost(gas_used: float, effective_gas_price: float) -> float:
    """Total cost of the transaction in wei: gas_used × effective_gas_price."""
    return gas_used * effective_gas_price


def calc_gas_efficiency(cumulative_gas_used: float, gas_used: float) -> float:
    """Dataset definition: cumulative block gas up to this tx ÷ this tx's gas.

    (Training mean ≈ 92; 0 whenever cumulative gas is unknown/0, which is the
    case for pending transactions.)
    """
    if gas_used <= 0:
        return 0.0
    return cumulative_gas_used / gas_used


def calc_gas_price_ratio(
    effective_gas_price: Optional[float],
    gas_price_cap: Optional[float],
) -> float:
    """Ratio of the effective (paid) gas price to the price the sender offered.

    In the training data this is 1.0 for virtually every transaction (mean
    0.99998, range [0, 1]), consistent with effective ÷ offered price. Returns
    1.0 when either value is unknown — e.g. for pending transactions, which have
    no effective price yet.
    """
    if not effective_gas_price or not gas_price_cap or gas_price_cap <= 0:
        return 1.0
    return effective_gas_price / gas_price_cap


# --------------------------------------------------------------------------- #
# Log / event metrics
# --------------------------------------------------------------------------- #

def calc_length_log(log_count: int) -> float:
    """Dataset definition: the number of log entries (integer count, max 43 in training)."""
    return float(log_count)


def calc_event_activity_flag(log_count: int) -> float:
    """1.0 if the transaction emitted ≥ 1 log event.

    Approximation: in the training data this agrees with "has logs" for ~82% of
    rows; the dataset's exact definition is not recoverable from its columns.
    """
    return 1.0 if log_count > 0 else 0.0


def calc_log_removed(logs: list) -> float:
    """1.0 if *any* log entry in the receipt has ``removed == True`` (chain reorg)."""
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
    """Character length of an address string (42 for a standard 0x address)."""
    return float(len(address)) if address else 0.0


def hash_length(tx_hash: Optional[str]) -> float:
    """Character length of the transaction hash string (66 for a standard hash)."""
    return float(len(tx_hash)) if tx_hash else 0.0
