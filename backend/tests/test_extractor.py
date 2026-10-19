# -*- coding: utf-8 -*-
"""
tests/test_extractor.py
=========================
Unit tests for the DeFiTransLyzer feature extractor.

Tests
------
- Standard eth_sendTransaction payload (full fields).
- eth_sendRawTransaction payload (hex-only).
- Receipt-style confirmed transaction payload.
- EIP-1559 fields (effectiveGasPrice, baseFeePerGas).
- Malformed / empty / None payloads.
- Individual metric helper functions.
"""
from __future__ import annotations

import math

import pytest

from feature_extractor.extractor import ExtractionError, extract_features
from feature_extractor.metrics import (
    calc_event_activity_flag,
    calc_gas_efficiency,
    calc_gas_price_ratio,
    calc_is_same_address,
    calc_length_log,
    calc_log_removed,
    calc_total_gas_cost,
)

# --------------------------------------------------------------------------- #
# Sample payloads
# --------------------------------------------------------------------------- #

ETH_SEND_TX = {
    "from": "0xAbCd1234abcd1234abcd1234abcd1234abcd1234",
    "to":   "0xEfGh5678efgh5678efgh5678efgh5678efgh5678",
    "gas":  "0x5208",       # 21000
    "gasPrice": "0x2e90edd000",  # 200 gwei
    "value": "0xde0b6b3a7640000",  # 1 ETH in wei
    "chainId": "0x1",
    "nonce": "0x5",
}

ETH_SEND_RAW_TX = {
    "params": ["0x" + "a" * 130]  # fake raw signed tx hex
}

RECEIPT_TX = {
    "hash": "0x" + "b" * 64,
    "from": "0xAbCd1234abcd1234abcd1234abcd1234abcd1234",
    "to":   "0xEfGh5678efgh5678efgh5678efgh5678efgh5678",
    "blockNumber": "0x1295820",
    "transactionIndex": "0x3",
    "gas": "0x5208",
    "gasUsed": "0x5208",
    "effectiveGasPrice": "0x2e90edd000",
    "cumulativeGasUsed": "0x7a120",
    "value": "0xde0b6b3a7640000",
    "chainId": "0x1",
    "logs": [
        {"removed": False, "data": "0x", "topics": []},
        {"removed": False, "data": "0x1234", "topics": []},
    ],
    "baseFeePerGas": "0x174876e800",  # 100 gwei
}

EIP1559_TX = {
    "from": "0xAbCd1234abcd1234abcd1234abcd1234abcd1234",
    "to":   "0xdeadbeef" + "0" * 32,
    "gas":  "0x7530",        # 30000
    "gasUsed": "0x7530",
    "maxFeePerGas": "0x3a35294400",       # 250 gwei offered
    "effectiveGasPrice": "0x2e90edd000",  # 200 gwei paid
    "baseFeePerGas": "0x174876e800",      # 100 gwei (not a model input)
    "chainId": "0x1",
    "value": "0x0",
    "logs": [],
}


# --------------------------------------------------------------------------- #
# Extractor — standard payloads
# --------------------------------------------------------------------------- #

class TestExtractFeaturesStandard:
    def test_eth_send_tx_returns_all_keys(self) -> None:
        features = extract_features(ETH_SEND_TX)
        expected_keys = {
            "length_transaction_hash", "length_to", "log_removed", "block_number",
            "gas_used", "length_from", "index", "gas_efficiency", "value",
            "chain_id", "total_gas_cost", "gas_per_log_event", "event_activity_flag",
            "normalized_token_transfer", "effective_gas_price", "cumulative_gas_used",
            "is_same_address", "gas_price_ratio", "length_log", "log_count",
        }
        assert expected_keys == set(features.keys())

    def test_all_values_are_float(self) -> None:
        features = extract_features(ETH_SEND_TX)
        for k, v in features.items():
            assert isinstance(v, float), f"Feature '{k}' is {type(v).__name__}, expected float"

    def test_length_to_correct(self) -> None:
        features = extract_features(ETH_SEND_TX)
        assert features["length_to"] == float(len(ETH_SEND_TX["to"]))

    def test_length_from_correct(self) -> None:
        features = extract_features(ETH_SEND_TX)
        assert features["length_from"] == float(len(ETH_SEND_TX["from"]))

    def test_is_same_address_zero_for_different_addresses(self) -> None:
        features = extract_features(ETH_SEND_TX)
        assert features["is_same_address"] == 0.0

    def test_chain_id_mainnet(self) -> None:
        features = extract_features(ETH_SEND_TX)
        assert features["chain_id"] == 1.0

    def test_no_logs_sets_event_flag_zero(self) -> None:
        features = extract_features(ETH_SEND_TX)
        assert features["event_activity_flag"] == 0.0
        assert features["log_count"] == 0.0
        assert features["length_log"] == 0.0

    def test_value_converted_from_hex(self) -> None:
        features = extract_features(ETH_SEND_TX)
        # 0xde0b6b3a7640000 = 1_000_000_000_000_000_000 (1 ETH in wei)
        assert features["value"] == pytest.approx(1e18)


class TestExtractFeaturesReceipt:
    def test_receipt_has_block_number(self) -> None:
        features = extract_features(RECEIPT_TX)
        assert features["block_number"] == float(int("0x1295820", 16))

    def test_receipt_log_features_use_dataset_definitions(self) -> None:
        features = extract_features(RECEIPT_TX)
        assert features["length_log"] == 2.0  # number of logs
        assert features["event_activity_flag"] == 1.0
        # Constant 0 in the training data
        assert features["log_count"] == 0.0
        assert features["gas_per_log_event"] == 0.0
        assert features["normalized_token_transfer"] == 0.0

    def test_receipt_gas_efficiency_is_cumulative_over_used(self) -> None:
        features = extract_features(RECEIPT_TX)
        assert features["gas_efficiency"] == pytest.approx(0x7A120 / 0x5208)

    def test_receipt_no_removed_logs(self) -> None:
        features = extract_features(RECEIPT_TX)
        assert features["log_removed"] == 0.0

    def test_receipt_with_removed_log(self) -> None:
        payload = dict(RECEIPT_TX)
        payload["logs"] = [{"removed": True}]
        features = extract_features(payload)
        assert features["log_removed"] == 1.0


class TestExtractFeaturesEIP1559:
    def test_gas_price_ratio_is_paid_over_offered(self) -> None:
        features = extract_features(EIP1559_TX)
        # paid 200 gwei / offered (maxFeePerGas) 250 gwei
        assert features["gas_price_ratio"] == pytest.approx(0.8, rel=1e-3)

    def test_pending_tx_gas_price_ratio_is_one(self) -> None:
        assert extract_features(ETH_SEND_TX)["gas_price_ratio"] == 1.0

    def test_total_gas_cost(self) -> None:
        features = extract_features(EIP1559_TX)
        gas_used = int("0x7530", 16)
        eff_price = int("0x2e90edd000", 16)
        assert features["total_gas_cost"] == pytest.approx(gas_used * eff_price, rel=1e-4)


# --------------------------------------------------------------------------- #
# Extractor — error handling
# --------------------------------------------------------------------------- #

class TestExtractFeaturesErrorHandling:
    def test_non_dict_raises_extraction_error(self) -> None:
        with pytest.raises(ExtractionError):
            extract_features("not a dict")  # type: ignore[arg-type]

    def test_empty_dict_returns_zero_features(self) -> None:
        features = extract_features({})
        # All features should be finite floats.
        # chain_id defaults to 1.0 (mainnet), gas_price_ratio defaults to 1.0
        # (no base_fee available), all others default to 0.0.
        for k, v in features.items():
            assert isinstance(v, float), f"Feature '{k}' is not a float"
            assert v >= 0.0, f"Feature '{k}' has unexpected negative value: {v}"

    def test_none_to_address_is_safe(self) -> None:
        payload = dict(ETH_SEND_TX)
        payload["to"] = None
        features = extract_features(payload)
        assert features["length_to"] == 0.0

    def test_malformed_hex_defaults_to_zero(self) -> None:
        payload = dict(ETH_SEND_TX)
        payload["gas"] = "not_hex"
        features = extract_features(payload)
        # Should not raise; gas_limit falls back to default 21000
        assert isinstance(features["gas_used"], float)


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #

class TestMetricHelpers:
    def test_total_gas_cost(self) -> None:
        assert calc_total_gas_cost(21_000, 200_000_000_000) == 4_200_000_000_000_000

    def test_gas_efficiency_zero_gas_used(self) -> None:
        assert calc_gas_efficiency(500_000, 0) == 0.0

    def test_gas_efficiency_cumulative_over_used(self) -> None:
        assert calc_gas_efficiency(42_000, 21_000) == 2.0

    def test_gas_price_ratio_unknown_paid_price(self) -> None:
        assert calc_gas_price_ratio(None, 200e9) == 1.0

    def test_length_log_is_count(self) -> None:
        assert calc_length_log(0) == 0.0
        assert calc_length_log(2) == 2.0

    def test_event_activity_flag(self) -> None:
        assert calc_event_activity_flag(0) == 0.0
        assert calc_event_activity_flag(1) == 1.0
        assert calc_event_activity_flag(5) == 1.0

    def test_log_removed_empty_logs(self) -> None:
        assert calc_log_removed([]) == 0.0

    def test_log_removed_with_removed(self) -> None:
        assert calc_log_removed([{"removed": True}]) == 1.0
        assert calc_log_removed([{"removed": False}, {"removed": True}]) == 1.0

    def test_is_same_address_case_insensitive(self) -> None:
        addr = "0xAbCd1234abcd1234abcd1234abcd1234abcd1234"
        assert calc_is_same_address(addr, addr.lower()) == 1.0
        assert calc_is_same_address(addr, "0xDifferent") == 0.0



class TestExtractFeaturesRobustness:
    """Malformed-but-dict payloads must degrade gracefully, never raise."""

    def test_logs_null_treated_as_empty(self) -> None:
        f = extract_features({"logs": None})
        assert f["length_log"] == 0.0

    def test_non_dict_log_entries_ignored(self) -> None:
        f = extract_features({"logs": [1, "junk", None, {"removed": True}]})
        assert f["length_log"] == 1.0
        assert f["log_removed"] == 1.0

    def test_logs_not_a_list_treated_as_empty(self) -> None:
        assert extract_features({"logs": "0x00"})["length_log"] == 0.0

    def test_non_string_addresses_coerced(self) -> None:
        f = extract_features({"from": 123, "to": 123})
        assert f["is_same_address"] == 1.0

    @pytest.mark.parametrize("value", ["1e999", "inf", "nan", float("inf")])
    def test_non_finite_values_default(self, value) -> None:
        f = extract_features({"value": value})
        assert f["value"] == 0.0
        assert all(math.isfinite(v) for v in f.values())
