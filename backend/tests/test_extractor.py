# -*- coding: utf-8 -*-
"""
tests/test_extractor.py
========================
Feature extraction from raw JSON-RPC payloads: payload shapes (pending, mixed,
tx+receipt, stored result pair), feature semantics and robustness to junk.
Feature definitions themselves live in models/features.py (shared with the
training repository).
"""
from __future__ import annotations

import math

import pytest

from feature_extractor.extractor import ExtractionError, extract_features, split_payload
from models.features import ALL_FEATURES, PRE_EXEC_FEATURES, TOPIC_TRANSFER

SENDER = "0x742d35cc6634c0532925a3b844bc454e4438f44e"
ROUTER = "0x7a250d5630b4cf539739df2c5dacb4c659f2488d"

ETH_SEND_TX = {
    "from": SENDER,
    "to": "0x9f44b3d4a44f91c9e7dc7b8e72f44f929d3c3bdd",
    "value": hex(10**18),
    "gas": hex(21000),
    "gasPrice": hex(20 * 10**9),
    "nonce": "0x0",
    "data": "0x",
}

EIP1559_TX = {
    "from": SENDER,
    "to": ROUTER,
    "value": "0x0",
    "gas": hex(250_000),
    "maxFeePerGas": hex(50 * 10**9),
    "nonce": "0x1f",
    "input": "0x38ed1739" + "00" * 32 + "ab" * 32,
}


def _transfer(token, frm, to, amount):
    pad = lambda a: "0x" + "00" * 12 + a[2:]
    return {"address": token, "topics": [TOPIC_TRANSFER, pad(frm), pad(to)], "data": hex(amount)}


RECEIPT = {
    "from": SENDER,
    "to": ROUTER,
    "gasUsed": hex(125_000),
    "status": "0x1",
    "contractAddress": None,
    "logs": [
        _transfer("0x" + "c" * 40, SENDER, "0x" + "d" * 40, 10**18),
        _transfer("0x" + "e" * 40, "0x" + "d" * 40, SENDER, 2 * 10**18),
    ],
}


class TestPayloadShapes:
    def test_pending_transaction(self) -> None:
        tx, receipt = split_payload(ETH_SEND_TX)
        assert tx is ETH_SEND_TX and receipt is None

    def test_tx_and_receipt_object(self) -> None:
        tx, receipt = split_payload({"tx": EIP1559_TX, "receipt": RECEIPT})
        assert tx is EIP1559_TX and receipt is RECEIPT

    def test_stored_rpc_result_pair(self) -> None:
        tx, receipt = split_payload({"result": [RECEIPT, EIP1559_TX, "Error!"]})
        assert tx is EIP1559_TX and receipt is RECEIPT

    def test_mixed_object_splits_off_receipt(self) -> None:
        tx, receipt = split_payload({**EIP1559_TX, "gasUsed": hex(100_000), "status": "0x0", "logs": []})
        assert receipt["gasUsed"] == hex(100_000)
        assert receipt["from"] == SENDER

    def test_non_dict_raises(self) -> None:
        with pytest.raises(ExtractionError):
            split_payload(["not", "a", "dict"])

    def test_bad_tx_object_raises(self) -> None:
        with pytest.raises(ExtractionError):
            split_payload({"tx": "0xdeadbeef"})


class TestPendingFeatures:
    def test_returns_every_feature_as_float(self) -> None:
        feats = extract_features(ETH_SEND_TX)
        assert list(feats) == ALL_FEATURES
        assert all(isinstance(v, float) for v in feats.values())

    def test_receipt_features_zero_without_receipt(self) -> None:
        feats = extract_features(EIP1559_TX)
        assert all(feats[n] == 0.0 for n in ALL_FEATURES if n not in PRE_EXEC_FEATURES)

    def test_value_gas_and_nonce(self) -> None:
        feats = extract_features(ETH_SEND_TX)
        assert math.expm1(feats["log_value_eth"]) == pytest.approx(1.0)
        assert math.expm1(feats["log_gas_price_gwei"]) == pytest.approx(20.0)
        assert math.expm1(feats["log_gas_limit"]) == pytest.approx(21000)
        assert feats["log_nonce"] == 0.0
        assert feats["has_calldata"] == 0.0

    def test_eip1559_uses_max_fee(self) -> None:
        feats = extract_features(EIP1559_TX)
        assert math.expm1(feats["log_gas_price_gwei"]) == pytest.approx(50.0)

    def test_selector_groups(self) -> None:
        feats = extract_features(EIP1559_TX)
        assert feats["sel_dex_swap"] == 1.0
        assert feats["sel_unknown"] == 0.0
        assert extract_features({**ETH_SEND_TX, "data": "0x095ea7b3"})["sel_approval"] == 1.0
        assert extract_features({**ETH_SEND_TX, "data": "0x12345678"})["sel_unknown"] == 1.0

    def test_calldata_zero_fraction(self) -> None:
        feats = extract_features(EIP1559_TX)
        assert feats["calldata_zero_frac"] == pytest.approx(32 / 68)

    def test_contract_creation_and_self_transfer(self) -> None:
        assert extract_features({**ETH_SEND_TX, "to": None})["is_contract_creation"] == 1.0
        same = extract_features({**ETH_SEND_TX, "to": SENDER.upper().replace("0X", "0x")})
        assert same["is_self_transfer"] == 1.0

    def test_address_identity_does_not_matter(self) -> None:
        other = {**ETH_SEND_TX, "from": "0x" + "1" * 40, "to": "0x" + "2" * 40}
        assert extract_features(other) == extract_features(ETH_SEND_TX)


class TestReceiptFeatures:
    def test_token_flows(self) -> None:
        feats = extract_features({"tx": EIP1559_TX, "receipt": RECEIPT})
        assert math.expm1(feats["log_n_transfers"]) == pytest.approx(2)
        assert feats["sender_sends_tokens"] == 1.0
        assert feats["sender_receives_tokens"] == 1.0
        assert feats["gas_used_ratio"] == pytest.approx(0.5)
        assert feats["status_failed"] == 0.0

    def test_failed_status(self) -> None:
        feats = extract_features({"tx": EIP1559_TX, "receipt": {**RECEIPT, "status": "0x0"}})
        assert feats["status_failed"] == 1.0

    def test_mint_from_zero_address(self) -> None:
        mint = _transfer("0x" + "c" * 40, "0x" + "0" * 40, SENDER, 5)
        feats = extract_features({"tx": EIP1559_TX, "receipt": {**RECEIPT, "logs": [mint]}})
        assert feats["has_mint"] == 1.0

    def test_block_position_fields_are_ignored(self) -> None:
        base = extract_features({"tx": EIP1559_TX, "receipt": RECEIPT})
        shifted = extract_features({
            "tx": {**EIP1559_TX, "blockNumber": "0xffffff", "transactionIndex": "0x9"},
            "receipt": {**RECEIPT, "blockNumber": "0xffffff", "cumulativeGasUsed": hex(29_000_000)},
        })
        assert shifted == base


class TestRobustness:
    @pytest.mark.parametrize("logs", [None, "abc", 5, [None, "junk", 3]], ids=str)
    def test_junk_logs_treated_as_empty(self, logs) -> None:
        feats = extract_features({"tx": EIP1559_TX, "receipt": {**RECEIPT, "logs": logs}})
        assert feats["log_n_logs"] == 0.0

    def test_junk_topics_ignored(self) -> None:
        bad = {"address": "0x" + "c" * 40, "topics": "not-a-list", "data": "0x"}
        feats = extract_features({"tx": EIP1559_TX, "receipt": {**RECEIPT, "logs": [bad]}})
        assert feats["log_n_transfers"] == 0.0
        assert math.expm1(feats["log_n_logs"]) == pytest.approx(1)

    @pytest.mark.parametrize("value", ["0xzz", "not-a-number", None, float("inf"), float("nan"), [1]], ids=str)
    def test_malformed_quantities_default_to_zero(self, value) -> None:
        feats = extract_features({**ETH_SEND_TX, "value": value})
        assert feats["log_value_eth"] == 0.0
        assert all(math.isfinite(v) for v in feats.values())

    def test_non_string_addresses_and_data(self) -> None:
        feats = extract_features({**ETH_SEND_TX, "from": 123, "to": ["x"], "data": 42})
        assert all(math.isfinite(v) for v in feats.values())
        assert feats["has_calldata"] == 0.0

    def test_empty_payload(self) -> None:
        feats = extract_features({})
        assert all(math.isfinite(v) for v in feats.values())
