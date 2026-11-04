# -*- coding: utf-8 -*-
"""Decoding signed raw transactions (eth_sendRawTransaction params)."""
from __future__ import annotations

import pytest
from eth_account import Account
from eth_utils import to_checksum_address

from inference_engine.services.raw_transaction import RawTransactionError, decode_raw_transaction

TO = to_checksum_address("0x8c5d7e12f3a4b5c6d7890abcdef3456789012cd3")
SEPOLIA = 11155111
ACCOUNT = Account.from_key("0x" + "11" * 32)


def sign(**fields):
    signed = ACCOUNT.sign_transaction(fields)
    return "0x" + bytes(signed.raw_transaction).hex(), "0x" + bytes(signed.hash).hex()


def test_eip1559_transfer():
    raw, tx_hash = sign(to=TO, value=10**17, gas=21000, maxFeePerGas=30 * 10**9,
                        maxPriorityFeePerGas=10**9, nonce=7, chainId=SEPOLIA)
    tx = decode_raw_transaction(raw)
    assert tx["type"] == "0x2"
    assert tx["from"] == ACCOUNT.address.lower()
    assert tx["to"] == TO.lower()
    assert int(tx["value"], 16) == 10**17
    assert int(tx["gas"], 16) == 21000
    assert int(tx["maxFeePerGas"], 16) == 30 * 10**9
    assert int(tx["nonce"], 16) == 7
    assert int(tx["chainId"], 16) == SEPOLIA
    assert tx["input"] == "0x"
    assert tx["hash"] == tx_hash


def test_legacy_eip155_transfer_with_calldata():
    data = "0xa9059cbb" + "00" * 64
    raw, tx_hash = sign(to=TO, value=0, gas=60000, gasPrice=2 * 10**9, nonce=0, chainId=SEPOLIA, data=data)
    tx = decode_raw_transaction(raw)
    assert tx["type"] == "0x0"
    assert int(tx["gasPrice"], 16) == 2 * 10**9
    assert int(tx["chainId"], 16) == SEPOLIA  # from v (EIP-155)
    assert tx["input"] == data
    assert tx["from"] == ACCOUNT.address.lower()
    assert tx["hash"] == tx_hash


def test_access_list_transaction():
    raw, _ = sign(to=TO, value=1, gas=30000, gasPrice=10**9, nonce=1, chainId=SEPOLIA, accessList=[])
    tx = decode_raw_transaction(raw)
    assert tx["type"] == "0x1"
    assert tx["from"] == ACCOUNT.address.lower()


def test_contract_creation_has_no_recipient():
    raw, _ = sign(value=0, gas=500000, maxFeePerGas=10**9, maxPriorityFeePerGas=10**9,
                  nonce=2, chainId=SEPOLIA, data="0x6080604052")
    assert decode_raw_transaction(raw)["to"] is None


def test_accepts_hex_without_prefix():
    raw, _ = sign(to=TO, value=1, gas=21000, gasPrice=10**9, nonce=0, chainId=SEPOLIA)
    assert decode_raw_transaction(raw[2:])["from"] == ACCOUNT.address.lower()


@pytest.mark.parametrize("raw", [None, 42, "", "0x", "0xzz", "0x7f00", "0x02c0", "0xc0", "0x02" + "ff" * 4])
def test_malformed_input_is_rejected(raw):
    with pytest.raises(RawTransactionError):
        decode_raw_transaction(raw)


def test_oversized_input_is_rejected():
    with pytest.raises(RawTransactionError, match="exceeds"):
        decode_raw_transaction("0x" + "00" * (1024 * 1024 + 1))
